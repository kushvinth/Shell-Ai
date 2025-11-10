import os
from PIL import Image
import pytesseract

# Set this if tesseract is not in your PATH (only for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_images_in_directory(directory_path, output_file='ocr_output.txt'):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    ocr_results = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {image_path}...")

            try:
                text = pytesseract.image_to_string(Image.open(image_path))
                ocr_results.append(f"--- {filename} ---\n{text}\n")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(ocr_results)
    
    print(f"OCR completed. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    directory = "OCR-NEW"  # Replace with your image directory
    ocr_images_in_directory(directory)
