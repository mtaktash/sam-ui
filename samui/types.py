from enum import Enum

from pydantic import BaseModel


class MouseButtons(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class Click(BaseModel):
    x: int
    y: int
    button: MouseButtons
