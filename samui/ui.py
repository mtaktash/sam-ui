import logging
from collections import defaultdict

import cv2
import numpy as np
from samui.types import Click, MouseButtons

BASE_SIZE = 512

logger = logging.getLogger(__name__)


def make_empty_frame(width: int = BASE_SIZE, height: int = BASE_SIZE) -> np.ndarray:
    return np.zeros((width, height, 3), dtype=np.uint8)


class SegmentAnythingUI:
    def __init__(self, winname: str, scale: float = 1.0):
        self._click_handlers = list()

        self._key_bindings = dict()
        self._key_handlers = defaultdict(set)

        self._rendering_pipeline = []
        self._scale = scale

        self._winname = winname
        cv2.namedWindow(self._winname, cv2.WINDOW_GUI_NORMAL)

    def set_rendering_pipeline(self, rendering_pipeline):
        self._rendering_pipeline = rendering_pipeline

    def render(self):
        frame = make_empty_frame()
        for render_step in self._rendering_pipeline:
            frame = render_step(frame)

        frame = cv2.resize(
            frame,
            (
                int(self._scale * frame.shape[1] + 0.5),
                int(self._scale * frame.shape[0] + 0.5),
            ),
        )
        window_size = cv2.getWindowImageRect(self._winname)
        if window_size[2] != frame.shape[1] or window_size[3] != frame.shape[0]:
            cv2.resizeWindow(self._winname, frame.shape[1], frame.shape[0])
        cv2.imshow(self._winname, frame)

    def process_inputs(self):
        raw_pressed_key = cv2.waitKeyEx(1)
        if raw_pressed_key != -1:
            pressed_key = self._key_bindings.get(raw_pressed_key, raw_pressed_key)
            if pressed_key != raw_pressed_key:
                logger.debug(f"key-press: {raw_pressed_key} -> {pressed_key}")
            else:
                logger.debug(f"key-press: {pressed_key}")

            if pressed_key in self._key_handlers:
                for handler in self._key_handlers[pressed_key]:
                    handler()

            if pressed_key == ord("q"):
                return False

        return True

    def run(self):
        cv2.setMouseCallback(self._winname, self.mouse_callback)

        stop = False
        while not stop:
            self.render()
            stop = not self.process_inputs()

    def mouse_callback(self, event, x, y, flags, param):
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            click = Click(
                x=int(x / self._scale + 0.5),
                y=int(y / self._scale + 0.5),
                button=(
                    MouseButtons.LEFT
                    if event == cv2.EVENT_LBUTTONDOWN
                    else MouseButtons.RIGHT
                ),
            )

            logger.debug(f"click: {click}")

            for handler in self._click_handlers:
                apply_next_handler = handler(click)
                if not apply_next_handler:
                    break

            return

    def bind_key(self, source_key: int, target_key: int):
        self._key_bindings[source_key] = target_key

    def add_key_handler(self, key, handler):
        self._key_handlers[key].add(handler)

    def remove_key_handler(self, key, handler):
        self._key_handlers[key].remove(handler)

    def add_click_handler(self, handler):
        self._click_handlers.append(handler)

    def remove_click_handler(self, handler):
        self._click_handlers.remove(handler)

    def show_image(self, image: np.ndarray):
        cv2.imshow(self._winname, image)

    def wait_key(self, delay: int = 1) -> int:
        return cv2.waitKeyEx(delay)

    def destroy(self):
        cv2.destroyWindow(self._winname)

    @property
    def scale(self):
        return self._scale
