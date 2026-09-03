import cv2
import numpy as np

from perspective import four_point_transform


IMAGE_PATH = "data/input/sample1.jpeg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Image not found")
    exit()

print("✅ Image loaded")


# --------------------------------
# 1. Resize image
# --------------------------------

height, width = image.shape[:2]

scale = 800 / width

resized = cv2.resize(
    image,
    (800, int(height * scale))
)

gray = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2GRAY
)


# --------------------------------
# 2. Blur
# --------------------------------

blur = cv2.GaussianBlur(
    gray,
    (7, 7),
    0
)


# --------------------------------
# 3. Threshold bright paper
# --------------------------------

_, threshold = cv2.threshold(
    blur,
    150,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    "data/output/page_threshold.jpeg",
    threshold
)

print("✅ Paper threshold created")


# --------------------------------
# 4. Morphological closing
# --------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (15, 15)
)

closed = cv2.morphologyEx(
    threshold,
    cv2.MORPH_CLOSE,
    kernel
)

cv2.imwrite(
    "data/output/page_closed.jpeg",
    closed
)

print("✅ Morphological processing completed")


# --------------------------------
# 5. Find external contours
# --------------------------------

contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# --------------------------------
# 6. Sort by area
# --------------------------------

contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)


print("Number of contours:", len(contours))


# --------------------------------
# 7. Find booklet boundary
# --------------------------------

page = None

image_area = resized.shape[0] * resized.shape[1]

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore small objects
    if area < image_area * 0.30:
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approximation) == 4:

        page = approximation

        print("✅ Possible booklet detected")
        print("Area:", area)

        break


# --------------------------------
# 8. Check detection
# --------------------------------

if page is None:

    print("❌ Could not detect booklet page")

    exit()


# --------------------------------
# 9. Display corners
# --------------------------------

print("Corners:")

for point in page:
    print(point[0])


# --------------------------------
# 10. Draw boundary
# --------------------------------

detected = resized.copy()

cv2.drawContours(
    detected,
    [page],
    -1,
    (0, 255, 0),
    4
)

cv2.imwrite(
    "data/output/detected_page.jpeg",
    detected
)

print("✅ Detection image saved")


# --------------------------------
# 11. Perspective correction
# --------------------------------

points = page.reshape(4, 2)

warped = four_point_transform(
    resized,
    points
)

cv2.imwrite(
    "data/output/warped.jpeg",
    warped
)

print("✅ Perspective corrected image saved")