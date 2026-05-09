import cv2
import os

print("Running preprocessing script...")
print("Current folder:", os.getcwd())

# load image
img = cv2.imread("test.jpg")

if img is None:
    print("❌ Image not found!")
else:
    print("✅ Image loaded")

    # resize
    img_resized = cv2.resize(img, (640, 640))

    # grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # denoise
    denoise = cv2.GaussianBlur(gray, (5,5), 0)

    # save image
    cv2.imwrite("preprocessed_image.jpg", denoise)

    print("✅ Saved as preprocessed_image.jpg")