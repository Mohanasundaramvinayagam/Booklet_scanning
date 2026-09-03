import cv2

from lighting import correct_lighting


INPUT_IMAGE = "data/output/00_standardized.jpg"
OUTPUT_IMAGE = "data/output/06_lighting_corrected.jpg"


image = cv2.imread(INPUT_IMAGE)

if image is None:
    print("❌ Could not load standardized image")
    exit()

print("✓ Standardized image loaded")

corrected = correct_lighting(image)

cv2.imwrite(
    OUTPUT_IMAGE,
    corrected
)

print("✓ Lighting correction completed")
print("Saved:", OUTPUT_IMAGE)