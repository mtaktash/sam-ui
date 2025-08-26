import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from samui.sam import SAMState, propagate_in_whole_video
from samui.types import Click, MouseButtons
from samui.ui import SegmentAnythingUI
from samui.utils import (
    load_video_frames,
    open_image,
    put_text,
    target_size_from_min_dimension,
)
from tqdm import tqdm

ARROW_UP = 63232
ARROW_DOWN = 63233
ARROW_LEFT = 63234
ARROW_RIGHT = 63235

ARROW_UP_LINUX = 65362
ARROW_DOWN_LINUX = 65364
ARROW_LEFT_LINUX = 65361
ARROW_RIGHT_LINUX = 65363

CLICK_RADIUS = 5

logger = logging.getLogger(__name__)


def get_cmap_color(idx):
    cmap = plt.get_cmap("tab20")
    rgba_color = cmap(idx % 20)
    return tuple(int(255 * x) for x in rgba_color[2::-1])


class TrackingUIState(BaseModel):
    frames: list[np.ndarray] | None = None
    current_frame_idx: int = 0

    clicks: dict[int, dict[int, list[Click]]] = Field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list[Click]))
    )  # object_id -> frame_idx -> clicks

    object_masks_by_frame: dict[tuple[int, int], np.ndarray] = Field(
        default_factory=dict
    )

    object_indices: set[int] = Field(default_factory=set)
    current_object_idx: int = 0

    output_path: str = "output/test"

    propagation_in_progress: bool = False
    temp_to_original_filenames: dict[str, str] = Field(default_factory=dict)

    test: bool = False

    class Config:
        arbitrary_types_allowed = True

    def reset_state(self):
        self.clicks.clear()
        self.object_masks_by_frame.clear()
        self.object_indices.clear()
        self.current_object_idx = 0


def get_ui_state(init_value: TrackingUIState | None = None):
    if not hasattr(get_ui_state, "state"):
        get_ui_state.state = init_value or TrackingUIState()
    elif init_value is not None:
        raise ValueError(
            "get_ui_state must be called without arguments on subsequent calls"
        )

    return get_ui_state.state


def get_ui(frames_path: str | None = None, ui_scale: float = 1.0):
    if not hasattr(get_ui, "ui"):
        state = get_ui_state()
        if state.frames is None:
            if frames_path is None:
                raise ValueError(
                    "Frames path must be provided on the first call to get_ui"
                )
            frames = load_video_frames(frames_path)
            state.frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        ui = SegmentAnythingUI("SAM-2 Tracking", scale=ui_scale)
        get_ui.ui = ui

    return get_ui.ui


def get_sam_state():
    if not hasattr(get_sam_state, "sam"):
        get_sam_state.sam = SAMState()

    return get_sam_state.sam


def present_frame(frame):
    state = get_ui_state()

    frame = state.frames[state.current_frame_idx]
    frame_size = target_size_from_min_dimension(frame, 512)
    frame = cv2.resize(frame, frame_size)

    return frame


def show_frame_idx(frame):
    state = get_ui_state()

    sam_state = get_sam_state()
    text_parts = [f"{state.current_frame_idx + 1}/{len(state.frames)}"]

    if state.test:
        text_parts.append("[test]")

    if not sam_state.initialized:
        text_parts.append("[SAM not ready]")

    if state.propagation_in_progress:
        text_parts.append("[propagating]")

    frame_size = [frame.shape[1], frame.shape[0]]
    bottom_left_corner = (10, frame_size[1] - 10)
    put_text(frame, bottom_left_corner, " ".join(text_parts))

    return frame


def show_object_idx(frame):
    state = get_ui_state()

    top_left_corner = (10, 30)
    color = get_cmap_color(state.current_object_idx)
    put_text(frame, top_left_corner, f"Object {state.current_object_idx}", color)

    return frame


def show_object_masks(frame):
    state = get_ui_state()

    all_object_ids = set(state.clicks.keys())
    for object_id in all_object_ids:
        object_mask = state.object_masks_by_frame.get(
            (object_id, state.current_frame_idx)
        )

        if object_mask is not None:
            frame_size = [frame.shape[1], frame.shape[0]]
            object_mask = cv2.resize(object_mask, frame_size)
            color = np.array(get_cmap_color(object_id)[:3])
            frame[object_mask.astype(bool)] = (
                np.dstack([object_mask] * 3) * color
            ).astype(np.uint8)[object_mask.astype(bool)]

    return frame


def show_clicks(frame):
    state = get_ui_state()

    for object_clicks in state.clicks.values():
        for click in object_clicks[state.current_frame_idx]:
            color = (0, 255, 0) if click.button == MouseButtons.LEFT else (0, 0, 255)
            cv2.circle(frame, (click.x, click.y), CLICK_RADIUS, color, -1)

    return frame


def decrement_frame_idx():
    state = get_ui_state()
    state.current_frame_idx = max(0, state.current_frame_idx - 1)


def increment_frame_idx():
    state = get_ui_state()
    state.current_frame_idx = min(len(state.frames) - 1, state.current_frame_idx + 1)


def decrement_object_idx():
    state = get_ui_state()
    state.current_object_idx = max(0, state.current_object_idx - 1)


def increment_object_idx():
    state = get_ui_state()
    state.current_object_idx = state.current_object_idx + 1


def add_click_to_sam(click: Click):
    sam = get_sam_state()
    if not sam.initialized:
        return

    ui_state = get_ui_state()

    frame_idx = ui_state.current_frame_idx
    object_id = ui_state.current_object_idx

    def _add_click_impl():
        with sam.lock:
            try:
                _, out_obj_ids, out_mask_logits = get_sam_state().add_point(
                    click.x,
                    click.y,
                    frame_idx,
                    object_id,
                    positive=click.button == MouseButtons.LEFT,
                )
                out_masks = postprocess_logits(out_mask_logits).cpu().numpy()

                for out_obj_id, out_mask in zip(out_obj_ids, out_masks):
                    ui_state.object_masks_by_frame[(out_obj_id, frame_idx)] = out_mask
            except Exception as e:
                logging.error(e)

    sam.apply_async(_add_click_impl)


def register_click(click: Click):
    state = get_ui_state()
    state.clicks[state.current_object_idx][state.current_frame_idx].append(click)
    add_click_to_sam(click)
    return True


def set_object_idx(object_idx: int):
    state = get_ui_state()
    state.current_object_idx = object_idx


def postprocess_logits(logits):
    logits = torch.sigmoid(logits[:, 0]).to(dtype=torch.float32)
    logits = (logits > 0.5).to(dtype=torch.float32)
    return logits


def get_clicks_at_xy(x: float, y: float, candidate_clicks: list[Click]):
    return [
        click
        for click in candidate_clicks
        if (click.x - x) ** 2 + (click.y - y) ** 2 < CLICK_RADIUS**2
    ]


def remove_current_object_and_frame_clicks_at_xy(click: Click):
    state = get_ui_state()

    current_object_clicks = state.clicks[state.current_object_idx]
    current_frame_clicks = current_object_clicks[state.current_frame_idx]

    clicks_to_remove = get_clicks_at_xy(click.x, click.y, current_frame_clicks)
    if not clicks_to_remove:
        return True

    for click in clicks_to_remove:
        current_frame_clicks.remove(click)

    new_sam_points = []
    new_sam_labels = []
    for click in current_frame_clicks:
        new_sam_points.append([click.x, click.y])
        new_sam_labels.append(1 if click.button == MouseButtons.LEFT else 0)

    del state.object_masks_by_frame[(state.current_object_idx, state.current_frame_idx)]

    sam = get_sam_state()

    def _remove_clicks_impl():
        with sam.lock:
            _, out_obj_ids, out_mask_logits = sam.reapply_points_for_object_and_frame(
                new_sam_points,
                new_sam_labels,
                state.current_object_idx,
                state.current_frame_idx,
            )
            out_masks = postprocess_logits(out_mask_logits).cpu().numpy()
            for out_obj_id, out_mask in zip(out_obj_ids, out_masks):
                state.object_masks_by_frame[(out_obj_id, state.current_frame_idx)] = (
                    out_mask
                )

    sam.apply_async(_remove_clicks_impl)

    return False


def propagate_all():
    sam = get_sam_state()
    if not sam.initialized:
        return

    ui_state = get_ui_state()
    if ui_state.propagation_in_progress:
        logger.warning("Propagation already in progress")
        return

    def _impl():
        ui_state.propagation_in_progress = True
        try:
            frame_idx = ui_state.current_frame_idx

            window = 16
            total_frames = len(ui_state.frames)

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in propagate_in_whole_video(
                sam.predictor,
                sam.state,
                start_frame_idx=frame_idx,
                window=window,
                total_frames=total_frames,
                reverse=False,
                lock=sam.lock,
            ):
                out_masks = postprocess_logits(out_mask_logits).cpu().numpy()

                for out_obj_id, out_mask in zip(out_obj_ids, out_masks):
                    ui_state.object_masks_by_frame[(out_obj_id, out_frame_idx)] = (
                        out_mask
                    )
        finally:
            ui_state.propagation_in_progress = False

    sam.apply_async(_impl)


def save_everything():
    state = get_ui_state()

    output_path = state.output_path
    os.makedirs(output_path, exist_ok=True)

    mask_output_path = os.path.join(output_path, "masks")
    click_output_path = os.path.join(output_path, "clicks")

    os.makedirs(mask_output_path, exist_ok=True)
    os.makedirs(click_output_path, exist_ok=True)

    for object_id, object_clicks in state.clicks.items():
        object_clicks_output_path = os.path.join(click_output_path, f"{object_id}")
        os.makedirs(object_clicks_output_path, exist_ok=True)

        for frame_idx, clicks in object_clicks.items():
            click_records = []
            for idx, click in enumerate(clicks):
                click_records.append(
                    f"{click.x} {click.y} {click.button == MouseButtons.LEFT}"
                )

            if not click_records:
                continue

            original_file_stem = state.temp_to_original_filenames.get(
                f"{frame_idx:04d}.jpg"
            )

            frame_clicks_output_path = os.path.join(
                object_clicks_output_path, f"{original_file_stem}.txt"
            )
            with open(frame_clicks_output_path, "w") as f:
                f.write("\n".join(click_records))

    sam = get_sam_state()
    with sam.lock:
        for (object_id, frame_idx), mask in state.object_masks_by_frame.items():

            if (mask > 0.5).sum() == 0:
                continue

            original_file_stem = state.temp_to_original_filenames.get(
                f"{frame_idx:04d}.jpg"
            )
            frame_mask_output_path = os.path.join(
                mask_output_path, f"{object_id}", f"{original_file_stem}.png"
            )
            os.makedirs(os.path.dirname(frame_mask_output_path), exist_ok=True)
            cv2.imwrite(frame_mask_output_path, (mask * 255).astype(np.uint8))


def load_everything(original_filenames_to_order: dict[str, int]):
    state = get_ui_state()

    output_path = state.output_path
    click_output_path = os.path.join(output_path, "clicks")

    for object_id in os.listdir(click_output_path):
        if object_id.startswith("."):
            continue
        object_clicks_output_path = os.path.join(click_output_path, object_id)
        for click_file in os.listdir(object_clicks_output_path):
            frame_idx = original_filenames_to_order.get(click_file.split(".")[0])

            if frame_idx is None:
                continue

            with open(os.path.join(object_clicks_output_path, click_file), "r") as f:
                for line in f:
                    x, y, is_left = line.strip().split(" ")
                    is_left = is_left == "True"
                    click = Click(
                        x=int(x),
                        y=int(y),
                        button=(MouseButtons.LEFT if is_left else MouseButtons.RIGHT),
                    )
                    state.clicks[int(object_id)][frame_idx].append(click)


def clear_output():
    state = get_ui_state()
    shutil.rmtree(state.output_path, ignore_errors=True)


def reset_state_completely():
    sam = get_sam_state()
    sam.reset_state()

    ui_state = get_ui_state()
    ui_state.reset_state()

    ui = get_ui()
    ui.render()


def init_sam(frames_path: str, sam: SAMState, state: TrackingUIState):
    sam.init_state(frames_path)
    for object_id in state.clicks.keys():
        for frame_idx in state.clicks[object_id].keys():
            for click in state.clicks[object_id][frame_idx]:
                add_click_to_sam(click)


def run_ui_for_split(
    split: list[int],
    jpeg_file_paths: list[Path],
    temp_frames_path: list[Path],
    state: TrackingUIState,
    ui_scale: float = 1.0,
):
    temp_frames_path.mkdir(exist_ok=True)

    logger.info(
        f"Resizing frames to 512px minimum dimension and temporarily saving to {temp_frames_path}"
    )

    state.reset_state()

    original_filenames_to_order = {}
    for i, idx in tqdm(enumerate(split), "Copying frames"):
        path = jpeg_file_paths[idx]
        image = open_image(path)
        image = image.resize(target_size_from_min_dimension(image, 512))
        state.temp_to_original_filenames[f"{i:04d}.jpg"] = path.stem
        original_filenames_to_order[path.stem] = idx
        image.save(temp_frames_path / f"{i:04d}.jpg", quality=100)

    if Path(state.output_path).exists():
        load_everything(original_filenames_to_order)

    ui = get_ui(str(temp_frames_path), ui_scale=ui_scale)
    ui.render()

    sam = get_sam_state()

    try:
        sam.apply_async(init_sam, temp_frames_path, sam, state)

        ui.set_rendering_pipeline(
            [
                present_frame,
                show_object_masks,
                show_clicks,
                show_frame_idx,
                show_object_idx,
            ]
        )

        ui.add_key_handler(ARROW_DOWN, decrement_object_idx)
        ui.add_key_handler(ARROW_UP, increment_object_idx)
        ui.add_key_handler(ARROW_LEFT, decrement_frame_idx)
        ui.add_key_handler(ARROW_RIGHT, increment_frame_idx)

        ui.bind_key(ARROW_UP_LINUX, ARROW_UP)
        ui.bind_key(ARROW_DOWN_LINUX, ARROW_DOWN)
        ui.bind_key(ARROW_LEFT_LINUX, ARROW_LEFT)
        ui.bind_key(ARROW_RIGHT_LINUX, ARROW_RIGHT)

        ui.add_key_handler(ord("p"), propagate_all)
        ui.add_key_handler(ord("r"), reset_state_completely)

        ui.add_key_handler(ord("s"), save_everything)
        ui.add_key_handler(ord("c"), clear_output)

        for hotkey_idx in range(10):
            ui.add_key_handler(
                ord(str(hotkey_idx)), partial(set_object_idx, hotkey_idx)
            )

        ui.add_click_handler(remove_current_object_and_frame_clicks_at_xy)
        ui.add_click_handler(register_click)

        ui.run()

    finally:
        ui.destroy()
        sam.destroy()

        del get_ui.ui
        del get_sam_state.sam
        del get_ui_state.state

        shutil.rmtree(temp_frames_path)


def main(args):
    frames_path = args.frames_path
    frames_path = Path(frames_path)

    jpeg_file_paths = [
        path
        for path in frames_path.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg"]
    ]
    jpeg_file_paths.sort(key=lambda path: path.stem)

    if args.clear_output:
        shutil.rmtree(args.output_path, ignore_errors=True)

    if args.train_test_json is not None:

        with open(args.train_test_json, "r") as f:
            train_test_json = json.load(f)

        train_filenames = train_test_json["train"]
        test_filenames = train_test_json["test"]

    train_indices = []
    test_indices = []
    for idx, path in tqdm(enumerate(jpeg_file_paths)):

        if args.test_k > 0:
            if idx % args.test_k == 0:
                test_indices.append(idx)
            else:
                train_indices.append(idx)

        else:
            if path.name in test_filenames:
                test_indices.append(idx)
            elif path.name in train_filenames:
                train_indices.append(idx)

    logger.info(f"Train frames: {len(train_indices)}")
    logger.info(f"Test frames: {len(test_indices)}")

    if args.train:
        state = get_ui_state(TrackingUIState(output_path=args.output_path + "/train"))
        temp_frames_path = Path(frames_path).parent / "frames_temp_train"
        run_ui_for_split(
            train_indices,
            jpeg_file_paths,
            temp_frames_path,
            state,
            args.ui_scale,
        )

    if args.test:
        state = get_ui_state(TrackingUIState(output_path=args.output_path + "/test"))
        state.test = True
        temp_frames_path = Path(frames_path).parent / "frames_temp_test"

        run_ui_for_split(
            test_indices,
            jpeg_file_paths,
            temp_frames_path,
            state,
            args.ui_scale,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-path",
        type=str,
        help="Path to the directory containing the frames",
        default="/mnt/data/scannetpp/data/0a7cc12c0e/iphone/undistorted_images",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory",
        default="output/0a7cc12c0e",
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        help="Scale factor for the UI",
        default=1.0,
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear the output directory before starting",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Only UI for train frames",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only UI for test frames",
    )
    parser.add_argument(
        "--test-k",
        type=int,
        help="Mark every k-th frame as a test frame. k = 0 means no test frames",
        default=8,
    )
    parser.add_argument(
        "--train-test-json",
        type=str,
        help="Mark frames in json as a test frames.",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()
    main(args)
