import cv2
import numpy as np


def correct_lighting(image):
    """
    Correct uneven lighting while preserving
    text and handwritten marks.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Estimate the background illumination
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        25
    )

    # Avoid division by zero
    background = np.maximum(
        background,
        1
    )

    # Normalize illumination
    corrected = (
        gray.astype(np.float32)
        / background.astype(np.float32)
    ) * 255

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    return corrected