import os
import pytesseract
from PIL import Image

# Directory containing images
image_dir = 'OCR-NEW'
output_file = 'ocr_new_output.txt'

# Get all image files in the directory, sorted by filename
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
])

all_text = []

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    all_text.append(f"--- {img_name} ---\n{text}\n")

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(all_text)

print(f"OCR complete. Extracted text saved to {output_file}")