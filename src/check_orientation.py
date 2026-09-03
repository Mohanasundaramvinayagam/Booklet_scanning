import cv2

IMAGE_PATH = "data/output/00_standardized.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Could not load standardized image")
    exit()

height, width = image.shape[:2]

print("Image loaded successfully")
print("Width :", width)
print("Height:", height)

if height > width:
    print("✓ Portrait orientation detected")
else:
    print("⚠ Landscape orientation detected")

cv2.imwrite(
    "data/output/05_orientation_checked.jpg",
    image
)

print("✓ Orientation check completed")