import logging
from contextlib import nullcontext
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor_hf
from sam2.sam2_video_predictor import SAM2VideoPredictor

logger = logging.getLogger(__name__)


def _get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.debug(f"Using device: {device}")

    if device.type == "cuda":
        # # use bfloat16 for the entire notebook
        # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return device


def load_sam_predictor() -> SAM2VideoPredictor:
    sam2_checkpoint = "facebook/sam2.1-hiera-large"

    logger.debug(f"Loading SAM2 model from {sam2_checkpoint}")

    return build_sam2_video_predictor_hf(
        sam2_checkpoint,
        device=_get_torch_device(),
    )


@torch.inference_mode()
def propagate_in_whole_video(
    predictor: SAM2VideoPredictor,
    inference_state,
    start_frame_idx=None,
    window=None,
    total_frames=None,
    reverse=False,
    progress_bar=True,
    lock=None,
):
    """Propagate the input points across frames to track in the entire video."""
    predictor.propagate_in_video_preflight(inference_state)

    obj_ids = inference_state["obj_ids"]
    batch_size = predictor._get_obj_num(inference_state)

    # set start index, end index, and processing order
    if start_frame_idx is None:
        # default: start from the earliest frame with input points
        start_frame_idx = min(
            t
            for obj_output_dict in inference_state["output_dict_per_obj"].values()
            for t in obj_output_dict["cond_frame_outputs"]
        )

    reverse_end_frame_idx = max(start_frame_idx - window, 0)
    if start_frame_idx > 0:
        reverse_processing_order = range(start_frame_idx, reverse_end_frame_idx - 1, -1)
    else:
        reverse_processing_order = []  # skip reverse tracking if starting from frame 0

    forward_end_frame_idx = min(start_frame_idx + window, total_frames - 1)
    forward_processing_order = range(start_frame_idx, forward_end_frame_idx + 1)

    if reverse:
        processing_order = list(forward_processing_order) + list(
            reverse_processing_order
        )
    else:
        processing_order = list(forward_processing_order)

    for frame_idx in tqdm(
        processing_order, desc="propagate in video", disable=not progress_bar
    ):
        lock_context = lock or nullcontext()

        with lock_context:
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if predictor.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        predictor._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = predictor._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=frame_idx < start_frame_idx,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": frame_idx < start_frame_idx
                }
                pred_masks_per_obj[obj_idx] = pred_masks

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]

            _, video_res_masks = predictor._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks


class SAMState:
    _predictor: SAM2VideoPredictor | None

    def __init__(self):
        self._predictor = None
        self._state = None
        self._pool = ThreadPool(2)
        self._lock = Lock()

    def init_state(self, frames_path: str):
        logger.debug("Initializing SAM state")
        if self._predictor is None:
            self.init_predictor()

        self._state = self._predictor.init_state(str(frames_path))

        self._state["num_pathway"] = 3
        self._state["iou_thre"] = 0.1
        self._state["uncertainty"] = 2

        logger.debug("SAM state initialized")

    def init_predictor(self):
        logger.debug("Initializing SAM predictor")
        self._predictor = load_sam_predictor()
        logger.debug("SAM predictor initialized")

    def reset_state(self):
        self._predictor.reset_state(
            self._state
        )  # this actually modifies the state in place!
        logger.debug("SAM state reset")

    def destroy(self):
        self._pool.close()
        logger.warning(
            "Destroying SAM and NOT joining SAM thread pool, some inferences might be lost"
        )
        # self._pool.join()

    def apply_async(self, func, *args, **kwargs):
        return self._pool.apply_async(func, args, kwargs)

    def add_point(
        self,
        x,
        y,
        frame_idx: int = 0,
        object_id: int = 0,
        positive: bool = True,
    ):
        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1 if positive else 0], np.int32)
        return self._predictor.add_new_points_or_box(
            inference_state=self._state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
            clear_old_points=False,
        )

    def reapply_points_for_object_and_frame(
        self,
        points: list[list[int]],
        labels: list[int],
        object_id: int,
        frame_idx: int,
    ):
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, np.int32)
        return self._predictor.add_new_points_or_box(
            inference_state=self._state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
            clear_old_points=True,
        )

    @property
    def predictor(self):
        return self._predictor

    @property
    def initialized(self):
        return self._state is not None

    @property
    def pool(self):
        return self._pool

    @property
    def state(self):
        return self._state

    @property
    def lock(self):
        return self._lock
