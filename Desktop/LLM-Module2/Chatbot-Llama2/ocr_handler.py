# import pytesseract
# from PIL import Image
# import cv2
# import fitz  # PyMuPDF
# import numpy as np

# # Set Tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# def extract_text_with_ocr(image_path):
#     """
#     Extract text from image files using OCR.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Apply preprocessing techniques
#     _, binary_image = cv2.threshold(
#         image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
#     )
#     denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 10, 7, 21)

#     # Perform OCR
#     text = pytesseract.image_to_string(Image.fromarray(denoised_image))
#     return text.strip()

# def extract_text_from_pdf_ocr(pdf_path):
#     """
#     Extract text from PDF files using OCR for each page.
#     """
#     extracted_text = ""

#     with fitz.open(pdf_path) as pdf_document:
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)

#             # Convert PDF page to grayscale image
#             image_bytes = page.get_pixmap().tobytes()
#             image = np.frombuffer(image_bytes, np.uint8)
#             image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

#             # Apply preprocessing techniques
#             _, binary_image = cv2.threshold(
#                 image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
#             )
#             denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 10, 7, 21)

#             # Perform OCR
#             page_text = pytesseract.image_to_string(Image.fromarray(denoised_image))
#             extracted_text += page_text + "\n"

#     return extracted_text.strip()
