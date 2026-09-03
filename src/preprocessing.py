import cv2
import os


def resize_image(image, target_height=1600):
    """
    Resize image while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    scale = target_height / height

    new_width = int(width * scale)

    resized = cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def preprocess_image(input_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------
    # Load image
    # --------------------------------

    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    print("Original image loaded")

    print(
        "Original size:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    # --------------------------------
    # Step 5: Resize / Standardize
    # --------------------------------

    resized = resize_image(image)

    cv2.imwrite(
        os.path.join(
            output_dir,
            "00_standardized.jpg"
        ),
        resized
    )

    print(
        "✓ Standardized:",
        resized.shape[1],
        "x",
        resized.shape[0]
    )

    # --------------------------------
    # Step 6: Grayscale
    # --------------------------------

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "01_grayscale.jpg"
        ),
        gray
    )

    print("✓ Grayscale completed")

    # --------------------------------
    # Step 7: Noise reduction
    # --------------------------------

    denoised = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "02_denoised.jpg"
        ),
        denoised
    )

    print("✓ Noise reduction completed")

    # --------------------------------
    # Step 8: Contrast enhancement
    # --------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "03_contrast.jpg"
        ),
        enhanced
    )

    print("✓ Contrast enhancement completed")

    # --------------------------------
    # Step 9: Adaptive threshold
    # --------------------------------

    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "04_threshold.jpg"
        ),
        threshold
    )

    print("✓ Adaptive threshold completed")

    return {
        "original": image,
        "standardized": resized,
        "grayscale": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "threshold": threshold
    }