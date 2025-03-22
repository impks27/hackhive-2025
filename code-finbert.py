import os
import json
import re
import pdfplumber
import mailparser
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load FinBERT model for finance-specific classification
try:
    classifier = pipeline("zero-shot-classification", model="ProsusAI/finbert")
    logger.info("FinBERT classifier loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load FinBERT model: {e}")
    raise

# Define request types and subcategories
REQUEST_TYPES = {
    "Money Movement - Inbound": {
        "description": "Funds received by the bank for transactions such as loan repayments or interest payments.",
        "subcategories": {
            "Principal": "Payment covering only the loan principal.",
            "Interest": "Payment covering accrued interest.",
            "Principal + Interest": "Payment covering both principal and interest.",
            "Principal + Interest + Fee": "Payment covering principal, interest, and fees."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Adjustment": {
        "description": "Modifications to existing financial agreements, including payment schedules, interest rates, or fees.",
        "fields": ["deal_name", "amount", "transaction_date"]
    }
}

# Regex patterns for extracting financial data
PATTERNS = {
    "deal_name": r"Deal Name[:\s]*([\w\s-]+)",
    "amount": r"Amount[:\s]*\$?([\d,]+\.?\d*)",
    "transaction_date": r"Transaction Date[:\s]*(\d{2}/\d{2}/\d{4})",
    "account_number": r"Account Number[:\s]*(\w+)"
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
    return text.strip()

# Function to extract text from EML
def extract_text_from_eml(eml_path):
    try:
        mail = mailparser.parse_from_file(eml_path)
        return mail.text_plain[0] if mail.text_plain else ""
    except Exception as e:
        logger.error(f"Error extracting text from {eml_path}: {e}")
        return ""

# Function to classify email content
def classify_email(content):
    if not content.strip():
        return {"request_type": "NA", "sub_request_type": "NA", "confidence": 0.0, "extracted_data": {}}
    
    main_result = classifier(content, list(REQUEST_TYPES.keys()), multi_label=False)
    top_request_type = main_result["labels"][0]
    confidence = main_result["scores"][0]
    
    subcategories = REQUEST_TYPES.get(top_request_type, {}).get("subcategories", {})
    if subcategories:
        sub_result = classifier(content, list(subcategories.keys()), multi_label=False)
        top_subcategory = sub_result["labels"][0]
    else:
        top_subcategory = "NA"
    
    extracted_data = {field: re.search(PATTERNS[field], content).group(1) if re.search(PATTERNS[field], content) else "Not Found" for field in REQUEST_TYPES.get(top_request_type, {}).get("fields", [])}
    
    return {"request_type": top_request_type, "sub_request_type": top_subcategory, "confidence": confidence, "extracted_data": extracted_data}

# Process all emails in a folder
def process_email_folder(folder_path):
    results = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            content = extract_text_from_pdf(file_path)
        elif file.endswith(".eml"):
            content = extract_text_from_eml(file_path)
        else:
            continue
        classification = classify_email(content)
        results.append({"file": file, "classification": classification})
    return results

# Main execution
if __name__ == "__main__":
    folder_path = "emails-new/"  # Update with the actual path
    results = process_email_folder(folder_path)
    print(json.dumps(results, indent=2))
