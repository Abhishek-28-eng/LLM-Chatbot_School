import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_with_ocr(pdf_path):
    """
    Extracts text from a PDF using OCR by converting PDF pages to images and then applying OCR.
    Args:
    - pdf_path: Path to the uploaded PDF file.
    
    Returns:
    - extracted_text: A string containing the extracted text from the PDF.
    """
    try:
        # Convert PDF to images (one per page)
        pages = convert_from_path(pdf_path, 300)  
        
        # Initialize the variable to hold the extracted text
        extracted_text = ""
        
        # Apply OCR to each page (convert image to text)
        for page_num, page in enumerate(pages):
            print(f"Processing page {page_num + 1}...")
            text = pytesseract.image_to_string(page)
            extracted_text += text
        
        return extracted_text
    
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None
