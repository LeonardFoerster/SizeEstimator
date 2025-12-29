import cv2
import os

img_path = os.path.abspath("test.jpg")
print(f"Trying to read: {img_path}")
img = cv2.imread(img_path)

if img is None:
    print("Failed to read image")
else:
    print(f"Successfully read image with shape: {img.shape}")
