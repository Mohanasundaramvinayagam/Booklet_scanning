import cv2
import os


INPUT_IMAGE = "data/output/roi_part_a.jpg"
OUTPUT_IMAGE = "data/output/part_a_handwriting_detection.jpg"


def detect_handwriting(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Threshold only for this experiment
    _, binary = cv2.threshold(
        gray,
        160,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    result = image.copy()

    detected = 0

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Ignore very small noise
        if area < 10:
            continue

        # Ignore extremely large regions
        if area > 2000:
            continue

        # Ignore very long table lines
        if w > 100 or h > 100:
            continue

        # Candidate handwritten component
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        detected += 1

    print("Detected components:", detected)

    return result


def main():

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    image = cv2.imread(
        INPUT_IMAGE
    )

    if image is None:
        print("❌ Could not load Part A ROI")
        return

    print("✓ Part A ROI loaded")

    result = detect_handwriting(
        image
    )

    success = cv2.imwrite(
        OUTPUT_IMAGE,
        result
    )

    if success:
        print("✓ Detection image saved")
        print("Saved:", OUTPUT_IMAGE)
    else:
        print("❌ Failed to save output")


if __name__ == "__main__":
    main()