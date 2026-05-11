"""
Step 2: PDF Text Extraction Module
====================================
This module handles extracting raw text from uploaded PDF resumes.
We use three methods for reliability:
  - pdfplumber: Best for text-based PDFs with complex layouts
  - PyPDF2: Fast, good for simple text PDFs
  - OCR (Tesseract): For image-based/scanned PDFs
"""

import io
import re
from PIL import Image
import pytesseract


def extract_text_with_pypdf2(pdf_file):
    """
    Extract text using PyPDF2.
    Good for: Standard text PDFs, simple layouts
    
    Args:
        pdf_file: File-like object or path to PDF
    Returns:
        str: Extracted text
    """
    try:
        import PyPDF2
        
        # Reset file pointer if needed
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        
        reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return '\n'.join(text_parts)
    
    except Exception as e:
        return f"PyPDF2 extraction failed: {str(e)}"


def extract_text_with_pdfplumber(pdf_file):
    """
    Extract text using pdfplumber.
    Good for: Complex layouts, tables, better spacing preservation
    
    Args:
        pdf_file: File-like object or path to PDF
    Returns:
        str: Extracted text
    """
    try:
        import pdfplumber
        
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        
        # Read file content into bytes buffer
        content = pdf_file.read() if hasattr(pdf_file, 'read') else open(pdf_file, 'rb').read()
        
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text_parts = []
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return '\n'.join(text_parts)
    
    except Exception as e:
        return f"pdfplumber extraction failed: {str(e)}"


def extract_text_with_ocr(pdf_file):
    """
    Extract text using OCR (Tesseract) for image-based PDFs.
    Good for: Scanned documents, image PDFs
    
    Args:
        pdf_file: File-like object or path to PDF
    Returns:
        str: Extracted text
    """
    try:
        # Check if Tesseract is available
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        return "OCR extraction failed: Tesseract not installed. Please install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki"
    
    try:
        import pdfplumber
        
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        
        content = pdf_file.read() if hasattr(pdf_file, 'read') else open(pdf_file, 'rb').read()
        
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Convert page to image
                page_image = page.to_image(resolution=300).original
                
                # OCR the image
                page_text = pytesseract.image_to_string(page_image)
                
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return '\n'.join(text_parts)
    
    except Exception as e:
        return f"OCR extraction failed: {str(e)}"


def clean_extracted_text(raw_text):
    """
    Clean and normalize extracted text.
    Removes artifacts from PDF extraction.
    
    Args:
        raw_text: Raw text from PDF extractor
    Returns:
        str: Cleaned text
    """
    if not raw_text:
        return ""
    
    # Remove page markers we added
    text = re.sub(r'--- Page \d+ ---\n', '', raw_text)
    
    # Fix common PDF extraction issues
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix broken words (hyphens at end of lines)
    text = re.sub(r'-\n(\w)', r'\1', text)
    
    # Normalize line breaks (max 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_resume_text(pdf_file):
    """
    Main extraction function: tries all three methods, picks best result.
    
    Strategy:
    1. Try pdfplumber first (best for text PDFs)
    2. Fallback to PyPDF2 (fast for simple PDFs)
    3. Fallback to OCR (for image-based PDFs)
    4. Pick whichever gives the most text
    
    Args:
        pdf_file: Uploaded PDF file object
    Returns:
        dict: {
            'text': str - extracted text,
            'method': str - which method worked,
            'char_count': int - character count,
            'success': bool
        }
    """
    results = {}
    
    # Try pdfplumber
    plumber_text = extract_text_with_pdfplumber(pdf_file)
    if not plumber_text.startswith('pdfplumber extraction failed'):
        results['pdfplumber'] = clean_extracted_text(plumber_text)
    
    # Try PyPDF2
    pypdf2_text = extract_text_with_pypdf2(pdf_file)
    if not pypdf2_text.startswith('PyPDF2 extraction failed'):
        results['pypdf2'] = clean_extracted_text(pypdf2_text)
    
    # Try OCR as last resort
    ocr_text = extract_text_with_ocr(pdf_file)
    if not ocr_text.startswith('OCR extraction failed'):
        results['ocr'] = clean_extracted_text(ocr_text)
    
    # Pick the result with most content
    if not results:
        return {
            'text': '',
            'method': 'none',
            'char_count': 0,
            'success': False,
            'error': 'All extraction methods failed'
        }
    
    # Prioritize: pdfplumber > PyPDF2 > OCR (if OCR gives reasonable text)
    best_method = None
    best_text = ''
    
    for method in ['pdfplumber', 'pypdf2', 'ocr']:
        if method in results and len(results[method]) > len(best_text):
            best_text = results[method]
            best_method = method
    
    return {
        'text': best_text,
        'method': best_method,
        'char_count': len(best_text),
        'success': len(best_text) > 50
    }
