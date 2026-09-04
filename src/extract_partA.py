import cv2
import os


INPUT_IMAGE = "data/output/roi_part_a.jpg"
OUTPUT_DIR = "data/output/part_a_questions"

DEBUG_IMAGE = "data/output/part_a_question_boundaries.jpg"


def main():

    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        print("❌ Could not load Part A ROI")
        return

    height, width = image.shape[:2]

    print("✓ Part A ROI loaded")
    print("Width :", width)
    print("Height:", height)

    # ------------------------------------------------
    # Q1-Q10 table area
    #
    # IMPORTANT:
    # These are proportions of the COMPLETE
    # Part A ROI.
    # ------------------------------------------------

    top = int(height * 0.233)
    bottom = int(height * 0.9)
    # Draw debug image
    debug = image.copy()

    # ------------------------------------------------
    # Divide the question area into 10 rows
    # ------------------------------------------------

    row_height = (bottom - top) / 10

    for question in range(10):

        y1 = int(
            top + question * row_height
        )

        y2 = int(
            top + (question + 1) * row_height
        )

        # Draw row boundary
        cv2.line(
            debug,
            (0, y1),
            (width - 1, y1),
            (0, 0, 255),
            2
        )

        # Question label
        cv2.putText(
            debug,
            f"Q{question + 1}",
            (5, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        print(
            f"Q{question + 1}:",
            y1,
            "to",
            y2
        )

    # Draw final boundary
    cv2.line(
        debug,
        (0, bottom),
        (width - 1, bottom),
        (0, 0, 255),
        2
    )

    # Save debug image
    cv2.imwrite(
        DEBUG_IMAGE,
        debug
    )

    print()
    print("✓ Debug image saved:")
    print(DEBUG_IMAGE)

    # ------------------------------------------------
    # Create question images
    # ------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for question in range(10):

        y1 = int(
            top + question * row_height
        )

        y2 = int(
            top + (question + 1) * row_height
        )

        # Small margin from horizontal borders
        margin = 2

        y1 += margin
        y2 -= margin

        question_image = image[
            y1:y2,
            :
        ]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"Q{question + 1}.jpg"
        )

        cv2.imwrite(
            output_path,
            question_image
        )

        print(
            f"✓ Q{question + 1} saved"
        )

    print()
    print("================================")
    print("QUESTION SPLITTING COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()