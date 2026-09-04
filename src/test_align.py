import cv2
import numpy as np
import os


TEMPLATE_PATH = "data/input/plain.jpeg"
TEST_IMAGE_PATH = "data/input/test5.jpeg"

OUTPUT_PATH = "data/output/aligned_test1.jpg"


def rotate_image(image, angle):
    """
    Rotate image by 0, 90, 180 or 270 degrees.
    """

    if angle == 0:
        return image

    if angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )


def prepare_image(image):
    """
    Convert image to grayscale and improve
    contrast for feature detection.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def find_best_alignment(template, test_image):

    template_gray = prepare_image(template)

    # SIFT is more robust than simple line detection
    sift = cv2.SIFT_create(
        nfeatures=4000
    )

    template_keypoints, template_descriptors = \
        sift.detectAndCompute(
            template_gray,
            None
        )

    if template_descriptors is None:
        raise RuntimeError(
            "Could not detect features in template"
        )

    matcher = cv2.BFMatcher()

    best_result = None

    # Test all major orientations.
    # This handles an upside-down phone image too.
    for angle in [0, 90, 180, 270]:

        rotated = rotate_image(
            test_image,
            angle
        )

        rotated_gray = prepare_image(
            rotated
        )

        test_keypoints, test_descriptors = \
            sift.detectAndCompute(
                rotated_gray,
                None
            )

        if test_descriptors is None:
            continue

        matches = matcher.knnMatch(
            template_descriptors,
            test_descriptors,
            k=2
        )

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            # Lowe ratio test
            if m.distance < 0.70 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            continue

        template_points = np.float32([
            template_keypoints[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        test_points = np.float32([
            test_keypoints[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        # Find transformation from test image
        # to template image
        homography, mask = cv2.findHomography(
            test_points,
            template_points,
            cv2.RANSAC,
            5.0
        )

        if homography is None or mask is None:
            continue

        inliers = int(
            mask.sum()
        )

        ratio = (
            inliers / len(good_matches)
        )

        result = {
            "angle": angle,
            "homography": homography,
            "good_matches": len(good_matches),
            "inliers": inliers,
            "ratio": ratio,
            "rotated_image": rotated
        }

        if (
            best_result is None
            or inliers > best_result["inliers"]
        ):
            best_result = result

    return best_result


def main():

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    template = cv2.imread(
        TEMPLATE_PATH
    )

    test_image = cv2.imread(
        TEST_IMAGE_PATH
    )

    if template is None:
        print("❌ Could not load template")
        print(TEMPLATE_PATH)
        return

    if test_image is None:
        print("❌ Could not load test image")
        print(TEST_IMAGE_PATH)
        return

    print("✓ Template loaded")
    print(
        "Template size:",
        template.shape[1],
        "x",
        template.shape[0]
    )

    print("✓ Test image loaded")
    print(
        "Test size:",
        test_image.shape[1],
        "x",
        test_image.shape[0]
    )

    print()
    print("Finding best alignment...")

    result = find_best_alignment(
        template,
        test_image
    )

    if result is None:

        print()
        print("❌ Alignment failed")
        print("Not enough reliable matches")
        return

    print()
    print("✓ Alignment found")

    print(
        "Detected rotation:",
        result["angle"],
        "degrees"
    )

    print(
        "Good matches:",
        result["good_matches"]
    )

    print(
        "Reliable matches:",
        result["inliers"]
    )

    print(
        "Inlier ratio:",
        round(result["ratio"], 3)
    )

    # Transform test image into
    # template coordinate system
    aligned = cv2.warpPerspective(
        result["rotated_image"],
        result["homography"],
        (
            template.shape[1],
            template.shape[0]
        )
    )

    success = cv2.imwrite(
        OUTPUT_PATH,
        aligned
    )

    if success:

        print()
        print("✓ Aligned image saved")
        print(
            "Saved:",
            OUTPUT_PATH
        )

    else:

        print("❌ Could not save aligned image")


if __name__ == "__main__":
    main()