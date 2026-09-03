import cv2
import numpy as np
import os


INPUT_IMAGE = "data/output/roi_register_number.jpg"
OUTPUT_DIR = "data/output/register_cells"


# --------------------------------
# 1. Load ROI
# --------------------------------

image = cv2.imread(
    INPUT_IMAGE,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("❌ Could not load register number ROI")
    exit()

print("✓ Register number ROI loaded")

height, width = image.shape

print("ROI size:", width, "x", height)


# --------------------------------
# 2. Threshold
# --------------------------------

binary = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    5
)


# --------------------------------
# 3. Detect vertical lines
# --------------------------------

vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, max(10, height // 2))
)

vertical_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel
)


cv2.imwrite(
    "data/output/register_vertical_lines.jpg",
    vertical_lines
)


# --------------------------------
# 4. Calculate vertical projection
# --------------------------------

projection = np.sum(
    vertical_lines > 0,
    axis=0
)


# --------------------------------
# 5. Find vertical line positions
# --------------------------------

line_positions = []

threshold_value = height * 0.25

inside_line = False
start = 0

for x, value in enumerate(projection):

    if value > threshold_value and not inside_line:
        start = x
        inside_line = True

    elif value <= threshold_value and inside_line:

        end = x

        center = (start + end) // 2

        line_positions.append(center)

        inside_line = False


# Handle final line
if inside_line:

    center = (start + width - 1) // 2

    line_positions.append(center)


print()
print("Detected vertical lines:")
print(line_positions)

print("Number of lines:", len(line_positions))


# --------------------------------
# 6. Remove very close detections
# --------------------------------

filtered_lines = []

for position in line_positions:

    if not filtered_lines:
        filtered_lines.append(position)

    elif position - filtered_lines[-1] > 5:
        filtered_lines.append(position)


line_positions = filtered_lines


print()
print("Filtered lines:")
print(line_positions)

print("Number of filtered lines:", len(line_positions))


# --------------------------------
# 7. Save detected-line visualization
# --------------------------------

visual = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2BGR
)

for x in line_positions:

    cv2.line(
        visual,
        (x, 0),
        (x, height),
        (0, 0, 255),
        2
    )


cv2.imwrite(
    "data/output/register_detected_lines.jpg",
    visual
)


print("✓ Line visualization saved")


# --------------------------------
# 8. Create cells
# --------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


cell_count = 0

for i in range(len(line_positions) - 1):

    x1 = line_positions[i]
    x2 = line_positions[i + 1]

    cell_width = x2 - x1

    # Ignore extremely small regions
    if cell_width < 15:
        continue

    # Small margin to avoid including box borders
    margin = 3

    left = x1 + margin
    right = x2 - margin

    if right <= left:
        continue

    cell = image[
        5:height - 5,
        left:right
    ]

    cell_count += 1

    filename = os.path.join(
        OUTPUT_DIR,
        f"cell_{cell_count:02d}.jpg"
    )

    cv2.imwrite(
        filename,
        cell
    )


print()
print("✓ Cells created:", cell_count)
print("Saved in:", OUTPUT_DIR)