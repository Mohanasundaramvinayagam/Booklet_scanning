import cv2
import os


INPUT_IMAGE = "data/output/aligned_test1.jpg"
OUTPUT_IMAGE = "data/output/roi_marks_table.jpg"


def extract_marks_table(image):
    """
    Extract the marks table from the template-aligned image.

    The coordinates are relative to the aligned template,
    not the original camera photograph.
    """

    height, width = image.shape[:2]

    # Template-relative ROI
    x1 = int(width * 0.06)
    y1 = int(height * 0.49)

    x2 = int(width * 0.96)
    y2 = int(height * 0.81)

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
        print("❌ Could not load aligned image")
        print(INPUT_IMAGE)
        return

    print("✓ Aligned image loaded")

    print(
        "Image size:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    roi = extract_marks_table(image)

    if roi.size == 0:
        print("❌ ROI is empty")
        return

    print(
        "ROI size:",
        roi.shape[1],
        "x",
        roi.shape[0]
    )

    success = cv2.imwrite(
        OUTPUT_IMAGE,
        roi
    )

    if success:
        print("✓ Marks table ROI saved")
        print("Saved:", OUTPUT_IMAGE)
    else:
        print("❌ Failed to save ROI")


if __name__ == "__main__":
    main()