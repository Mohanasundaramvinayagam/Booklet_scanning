import cv2
import os


def crop_register_number(image):

    # Register number region
    x1 = 330
    y1 = 295
    x2 = 1300
    y2 = 395

    roi = image[y1:y2, x1:x2]

    return roi


def main():

    input_path = "data/output/03_contrast.jpg"
    output_path = "data/output/roi_register_number.jpg"

    # Load image
    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print("❌ Could not load image")
        return

    print("✓ Image loaded")

    # Show actual image dimensions
    height, width = image.shape

    print("Image size:", width, "x", height)

    # Crop ROI
    roi = crop_register_number(image)

    # --------------------------------
    # Validate ROI
    # --------------------------------

    if roi.size == 0:
        print("❌ ROI is empty")
        print("Check the coordinates")
        return

    print(
        "ROI size:",
        roi.shape[1],
        "x",
        roi.shape[0]
    )

    # Create output directory
    os.makedirs(
        "data/output",
        exist_ok=True
    )

    # Save ROI
    success = cv2.imwrite(
        output_path,
        roi
    )

    if success:
        print("✓ Register Number ROI saved")
        print("Saved:", output_path)
    else:
        print("❌ Failed to save ROI")


if __name__ == "__main__":
    main()