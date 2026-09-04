import cv2
import numpy as np
import os
import csv


INPUT_IMAGE = "data/output/roi_part_a.jpg"

OUTPUT_DIR = "data/output/handwritten_marks"
CSV_FILE = "data/output/handwritten_marks.csv"


# ---------------------------------------------------
# Your confirmed Part A question layout
# ---------------------------------------------------

TOP = 0.233
BOTTOM = 0.90

NUMBER_OF_QUESTIONS = 10


def remove_table_lines(binary):

    # Remove horizontal table lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    binary = cv2.subtract(
        binary,
        horizontal
    )

    # Remove vertical table lines
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 15)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    binary = cv2.subtract(
        binary,
        vertical
    )

    return binary


def extract_mark(row):

    height, width = row.shape[:2]

    # ------------------------------------------------
    # Marks column
    # ------------------------------------------------

    x1 = int(width * 0.72)
    x2 = int(width * 0.98)

    mark_area = row[:, x1:x2]

    gray = cv2.cvtColor(
        mark_area,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce small camera noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # ------------------------------------------------
    # Threshold
    # ------------------------------------------------

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7
    )

    # ------------------------------------------------
    # Remove table lines
    # ------------------------------------------------

    binary = remove_table_lines(
        binary
    )

    # ------------------------------------------------
    # Find connected components
    # ------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_components = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        # Ignore very small noise
        if area < 12:
            continue

        if w < 4 or h < 4:
            continue

        # Ignore tiny square/point noise
        if w <= 6 and h <= 6:
            continue

        # Ignore horizontal table-line remnants
        if w > width * 0.40 and h < 6:
            continue

        # Ignore vertical table-line remnants
        if h > height * 0.75 and w < 6:
            continue

        # Ignore components touching row borders
        border_margin = 3

        if y <= border_margin:
            continue

        if y + h >= height - border_margin:
            continue

        valid_components.append(
            (x, y, w, h, area)
        )

    # ------------------------------------------------
    # No handwriting found
    # ------------------------------------------------

    if len(valid_components) == 0:

        return None

    # ------------------------------------------------
    # Find bounding box around handwriting
    # ------------------------------------------------

    x1 = min(
        c[0]
        for c in valid_components
    )

    y1 = min(
        c[1]
        for c in valid_components
    )

    x2 = max(
        c[0] + c[2]
        for c in valid_components
    )

    y2 = max(
        c[1] + c[3]
        for c in valid_components
    )

    # ------------------------------------------------
    # Padding
    # ------------------------------------------------

    padding = 4

    x1 = max(
        0,
        x1 - padding
    )

    y1 = max(
        0,
        y1 - padding
    )

    x2 = min(
        binary.shape[1],
        x2 + padding
    )

    y2 = min(
        binary.shape[0],
        y2 + padding
    )

    extracted = binary[
        y1:y2,
        x1:x2
    ]

    return extracted


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    image = cv2.imread(
        INPUT_IMAGE
    )

    if image is None:

        print(
            "❌ Could not load Part A ROI"
        )

        return

    height, width = image.shape[:2]

    print("✓ Part A ROI loaded")

    print(
        "Size:",
        width,
        "x",
        height
    )

    # ------------------------------------------------
    # Calculate Q1-Q10 rows
    # ------------------------------------------------

    top = int(
        height * TOP
    )

    bottom = int(
        height * BOTTOM
    )

    row_height = (
        bottom - top
    ) / NUMBER_OF_QUESTIONS

    results = []

    for question in range(
        1,
        NUMBER_OF_QUESTIONS + 1
    ):

        y1 = int(
            top +
            (question - 1) *
            row_height
        )

        y2 = int(
            top +
            question *
            row_height
        )

        # Small border margin
        y1 += 2
        y2 -= 2

        row = image[
            y1:y2,
            :
        ]

        extracted = extract_mark(
            row
        )

        if extracted is None:

            print(
                f"Q{question}: empty"
            )

            results.append([
                question,
                "",
                "empty"
            ])

            continue

        output_path = os.path.join(
            OUTPUT_DIR,
            f"Q{question}.png"
        )

        cv2.imwrite(
            output_path,
            extracted
        )

        print(
            f"Q{question}: handwriting extracted"
        )

        results.append([
            question,
            output_path,
            "detected"
        ])

    # ------------------------------------------------
    # CSV
    # ------------------------------------------------

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "question",
            "handwriting_image",
            "status"
        ])

        writer.writerows(
            results
        )

    print()
    print("================================")
    print("HANDWRITING EXTRACTION COMPLETE")
    print("================================")

    print(
        "Images:",
        OUTPUT_DIR
    )

    print(
        "CSV:",
        CSV_FILE
    )


if __name__ == "__main__":
    main()