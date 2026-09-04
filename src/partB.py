import cv2
import os


INPUT_IMAGE = "data/output/roi_marks_table.jpg"
OUTPUT_IMAGE = "data/output/roi_part_b.jpg"


def extract_part_b(image):
    """
    Extract the complete Part B region.

    This is based on the template-aligned marks-table image.
    We keep Part B as one large ROI instead of splitting
    individual question cells.
    """

    height, width = image.shape[:2]

    # Part B occupies the middle portion of the marks table.
    #
    # These are relative to roi_marks_table.jpg.
    x1 = int(width * 0.29)
    y1 = int(height * 0.00)

    x2 = int(width * 0.85)
    y2 = int(height * 0.86)

    roi = image[y1:y2, x1:x2]

    return roi


def main():

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    image = cv2.imread(
        INPUT_IMAGE
    )

    if image is None:
        print("❌ Could not load marks table")
        print(INPUT_IMAGE)
        return

    print("✓ Marks table loaded")

    print(
        "Image size:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    roi = extract_part_b(image)

    if roi.size == 0:
        print("❌ Part B ROI is empty")
        return

    print(
        "Part B ROI size:",
        roi.shape[1],
        "x",
        roi.shape[0]
    )

    success = cv2.imwrite(
        OUTPUT_IMAGE,
        roi
    )

    if success:
        print("✓ Part B ROI saved")
        print("Saved:", OUTPUT_IMAGE)
    else:
        print("❌ Failed to save Part B ROI")


if __name__ == "__main__":
    main()