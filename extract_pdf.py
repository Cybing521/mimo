import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, output_txt_path):
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        
        # Extract metadata if available
        if reader.metadata:
            text_content.append(f"Metadata: {reader.metadata}\n")
            
        # Extract text from each page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_content.append(f"--- Page {i+1} ---\n{text}\n")
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("".join(text_content))
            
        print(f"Successfully extracted text to {output_txt_path}")
        
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    pdf_file = "papers/Ma 等 - 2023 - MIMO Capacity Characterization for Movable Antenna-已解锁.pdf"
    output_file = "papers/ma_2023_extracted.txt"
    extract_text_from_pdf(pdf_file, output_file)

