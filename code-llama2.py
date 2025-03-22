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

# Load zero-shot classification model
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    logger.error(f"Failed to load classifier model: {e}")
    raise

# Request Type Definitions (Main & Subcategories)
REQUEST_TYPES = {
    "Money Movement - Inbound": {
        "description": "Any money coming into the bank, such as customer loan repayments, incoming wire transfers, and deposits.",
        "subcategories": {
            "Customer Loan Repayment": "A customer is making a payment towards their loan balance.",
            "Incoming Wire Transfer": "Funds received via wire transfer from another bank or financial institution.",
            "Deposit Received": "A deposit made into a customer account or loan payment."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Money Movement - Outbound": {
        "description": "Any money going out of the bank, such as loan disbursements, refunds, or wire transfers sent to customers.",
        "subcategories": {
            "Loan Disbursement": "Funds released to a borrower as part of a loan agreement.",
            "Customer Refund": "A refund issued to a customer due to overpayment or service issue.",
            "Wire Transfer Sent": "Funds sent to another account via wire transfer."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Billing Issue": {
        "description": "Customer inquiries related to incorrect charges, missing payments, or overcharges on accounts.",
        "subcategories": {
            "Incorrect Charge": "A customer disputes a charge on their account.",
            "Missing Payment": "A customer claims a payment was made but not credited.",
            "Overcharge": "A customer was charged more than expected."
        },
        "fields": ["invoice_number", "billing_date", "amount"]
    }
}

# Regex patterns for data extraction
PATTERNS = {
    "deal_name": r"Deal Name[:\s]*([\w\s-]+)",
    "amount": r"Amount[:\s]*\$?([\d,]+\.?\d*)",
    "transaction_date": r"Transaction Date[:\s]*(\d{2}/\d{2}/\d{4})",
    "invoice_number": r"Invoice Number[:\s]*(\w+)",
    "billing_date": r"Billing Date[:\s]*(\d{2}/\d{2}/\d{4})"
}

# --- Utility Functions ---

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            logger.info(f"Successfully extracted text from PDF: {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_eml(eml_path):
    """Extract text from an .eml email file."""
    try:
        mail = mailparser.parse_from_file(eml_path)
        text = f"Subject: {mail.subject}\n{mail.body}"
        logger.info(f"Successfully extracted text from EML: {eml_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {eml_path}: {e}")
        return ""

def extract_fields(content, request_type):
    """Extract specific fields from content based on request type using regex."""
    extracted_data = {}
    fields = REQUEST_TYPES.get(request_type, {}).get("fields", [])

    for field in fields:
        pattern = PATTERNS.get(field)
        if pattern:
            match = re.search(pattern, content, re.IGNORECASE)
            extracted_data[field] = match.group(1) if match else "Not Found"
    
    return extracted_data

# --- Classification Logic ---

def classify_email(content):
    """Classify email content into main and subcategories."""
    if not content.strip():
        logger.warning("Empty content provided for classification.")
        return {
            "request_type": "NA",
            "sub_request_type": "NA",
            "reason": "No meaningful content found.",
            "confidence": 0.0,
            "extracted_data": {}
        }

    # Prepare main request type classification
    main_request_types = list(REQUEST_TYPES.keys())
    main_request_descriptions = "\n".join([f"- {key}: {value['description']}" for key, value in REQUEST_TYPES.items()])

    main_prompt = f"""
    You are an AI email classifier for a Loan Services bank. Your job is to classify emails into predefined request types based on their content.

    Here are the request types and their meanings:
    {main_request_descriptions}

    Given the following email:
    ---
    {content}
    ---

    Classify the email into one of the request types listed above and provide a brief reasoning.
    """

    try:
        # Classify main request type
        main_result = classifier(main_prompt, main_request_types)
        top_main_request = main_result["labels"][0]
        main_confidence = main_result["scores"][0]
        main_reason = REQUEST_TYPES[top_main_request]["description"]

        # Check for subcategories
        sub_request_types = REQUEST_TYPES[top_main_request].get("subcategories", {})
        if sub_request_types:
            sub_request_labels = list(sub_request_types.keys())
            sub_request_descriptions = "\n".join([f"- {key}: {value}" for key, value in sub_request_types.items()])

            sub_prompt = f"""
            Now that the email has been classified as '{top_main_request}', identify the specific subcategory.

            Here are the subcategories:
            {sub_request_descriptions}

            Given the email:
            ---
            {content}
            ---

            Classify it into one of the subcategories listed above.
            """

            sub_result = classifier(sub_prompt, sub_request_labels)
            top_sub_request = sub_result["labels"][0]
            sub_confidence = sub_result["scores"][0]
            sub_reason = sub_request_types[top_sub_request]

            # Extract fields for the classified request type
            extracted_data = extract_fields(content, top_main_request)

            logger.info(f"Classified email as {top_main_request} - {top_sub_request} with confidence {min(main_confidence, sub_confidence):.4f}")
            return {
                "request_type": top_main_request,
                "sub_request_type": top_sub_request,
                "reason": sub_reason,
                "confidence": round(min(main_confidence, sub_confidence), 4),
                "extracted_data": extracted_data
            }

        # No subcategories case
        extracted_data = extract_fields(content, top_main_request)
        logger.info(f"Classified email as {top_main_request} with confidence {main_confidence:.4f}")
        return {
            "request_type": top_main_request,
            "sub_request_type": "NA",
            "reason": main_reason,
            "confidence": round(main_confidence, 4),
            "extracted_data": extracted_data
        }

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return {
            "request_type": "NA",
            "sub_request_type": "NA",
            "reason": "Classification error.",
            "confidence": 0.0,
            "extracted_data": {}
        }

# --- Main Processing Logic ---

def process_email_directory(directory):
    """Process all emails in a directory and classify them."""
    results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".eml"):
            text = extract_text_from_eml(file_path)
        else:
            logger.warning(f"Skipping unsupported file: {filename}")
            continue

        classification = classify_email(text)
        results.append({"file": filename, **classification})

    return results

# --- Execution ---

if __name__ == "__main__":
    # Directory containing email files
    EMAIL_DIRECTORY = "/Users/paramita.santra/impks/hackhive-2025/emails"
    
    # Process emails and get results
    classification_results = process_email_directory(EMAIL_DIRECTORY)

    # Save results to JSON
    OUTPUT_FILE = "classification_results.json"
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(classification_results, f, indent=2)
        logger.info(f"Classification completed! Results saved in {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results to {OUTPUT_FILE}: {e}")

    # Print results
    print("\n📌 Classification Results:\n")
    print(json.dumps(classification_results, indent=2))