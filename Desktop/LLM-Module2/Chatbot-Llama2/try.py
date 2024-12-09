from ocr_handler import extract_text_with_ocr

image_path = "data/images.jpg"  # Replace with your image path
extracted_text = extract_text_with_ocr(image_path)

print("Extracted Text:")
print(extracted_text)
