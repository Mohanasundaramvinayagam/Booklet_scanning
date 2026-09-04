import cv2
import os

INPUT_IMAGE = "data/output/roi_marks_table.jpg"
OUTPUT_IMAGE = "data/output/roi_part_a.jpg"


def extract_part_a(image):

    height, width = image.shape[:2]

    # Entire Part A region
    x1 = int(width * 0.00)
    y1 = int(height * 0.00)

    x2 = int(width * 0.29)
    y2 = int(height * 0.86)

    roi = image[y1:y2, x1:x2]

    return roi


def main():

    os.makedirs("data/output", exist_ok=True)

    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        print("❌ Could not load marks table")
        return

    print("✓ Marks table loaded")

    roi = extract_part_a(image)

    if roi.size == 0:
        print("❌ Part A ROI is empty")
        return

    cv2.imwrite(
        OUTPUT_IMAGE,
        roi
    )

    print("✓ Complete Part A region extracted")
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()