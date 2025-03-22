import os
import json
import re
import pdfplumber
import mailparser
from transformers import pipeline
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load zero-shot classification model
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.info("Classifier loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load classifier model: {e}")
    raise

# Request Type Definitions (Main & Subcategories)
REQUEST_TYPES = {
    # "Adjustment": {
    #     "description": "Revisions or modifications made to existing financial agreements, obligations, or fee structures.",
    #     "fields": ["deal_name", "amount", "transaction_date"]
    # },
    "AU Transfer": {
        "description": "Fund transfers related to Allocation Units (AU), where a principal amount is moved between different financial structures.",
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Closing Notice": {
        "description": "Notifications or actions related to terminating or modifying an existing financial agreement.",
        "subcategories": {
            "Reallocation Fees": "Charges incurred when reallocating funds, assets, or positions within an agreement.",
            "Amendment Fees": "Fees applied for modifications or contractual adjustments to the terms of an agreement.",
            "Reallocation Principal": "An adjustment to the principal amount during a reallocation process."
        },
        "fields": ["deal_name", "transaction_date", "amount"]
    },
    "Commitment Change": {
        "description": "Adjustments to the level of committed financial resources or obligations within a loan or credit facility.",
        "subcategories": {
            "Decrease": "A reduction in the committed amount or financial obligation.",
            "Increase": "An increase in the committed amount or financial obligation.",
            "Cashless Roll": "A transaction where an existing commitment is rolled over without cash settlement."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Fee Payment": {
        "description": "Payments related to fees associated with financial agreements or loan services.",
        "subcategories": {
            "Ongoing Fee": "Recurring fees charged for continuous services or loan maintenance.",
            "Letter of Credit Fee": "Fees associated with issuing or amending a letter of credit."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Money Movement - Inbound": {
        "description": "Transactions involving funds being received by the bank, such as loan repayments or interest payments.",
        "subcategories": {
            "Principal": "Payment covering only the original loan amount.",
            "Interest": "Payment covering accrued interest on a loan.",
            "Principal + Interest": "Combined payment of principal and interest.",
            "Principal + Interest + Fee": "Payment including principal, interest, and fees."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Money Movement - Outbound": {
        "description": "Transactions involving funds leaving the bank, such as loan disbursements or transfers.",
        "subcategories": {
            "Timebound": "Scheduled or deadline-driven fund transfer.",
            "Foreign Currency": "Outbound transaction in a different currency."
        },
        "fields": ["deal_name", "amount", "transaction_date", "currency"]
    }
}

# Regex patterns for data extraction
PATTERNS = {
    "deal_name": r"Deal Name[:\s]*([\w\s-]+)",
    "amount": r"Amount[:\s]*\$?([\d,]+\.?\d*)",
    "transaction_date": r"Transaction Date[:\s]*(\d{2}/\d{2}/\d{4})",
    "account_number": r"Account Number[:\s]*(\w+)",
    "currency": r"Currency[:\s]*([A-Z]{3})"
}

# --- Utility Functions ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            logger.info(f"Successfully extracted text from PDF: {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_eml(eml_path: str) -> str:
    """Extract text from an .eml email file."""
    try:
        mail = mailparser.parse_from_file(eml_path)
        text = f"Subject: {mail.subject}\n{mail.body}"
        logger.info(f"Successfully extracted text from EML: {eml_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {eml_path}: {e}")
        return ""

def extract_fields(content: str, request_type: str) -> Dict[str, str]:
    """Extract specific fields from content based on request type using regex."""
    extracted_data = {}
    fields = REQUEST_TYPES.get(request_type, {}).get("fields", [])

    for field in fields:
        pattern = PATTERNS.get(field)
        if pattern:
            matches = re.findall(pattern, content, re.IGNORECASE)
            extracted_data[field] = matches[0].strip() if matches else "Not Found"
    
    return extracted_data

# --- Classification Logic ---

def classify_email(content: str) -> List[Dict]:
    """Classify email content into multiple request types with primary intent detection."""
    if not content.strip():
        logger.warning("Empty content provided for classification.")
        return [{
            "request_type": "NA",
            "sub_request_type": "NA",
            "reason": "No meaningful content found.",
            "confidence": 0.0,
            "extracted_data": {},
            "is_primary": True
        }]

    # Improved segmentation: Split by double newlines or explicit request indicators
    segments = [seg.strip() for seg in re.split(r'\n{2,}|(?:Additionally|Also|Furthermore)[,\s]', content, flags=re.IGNORECASE) if seg.strip()]
    if not segments:
        segments = [content]  # Fallback to whole content

    results = []
    main_request_types = list(REQUEST_TYPES.keys())
    main_request_descriptions = "\n".join([f"- {key}: {value['description']}" for key, value in REQUEST_TYPES.items()])

    # Classify each segment
    for segment in segments:
        print("segment: ", segment)
        main_prompt = f"""
        You are an AI email classifier for a Loan Services bank. Classify this email segment into a request type based on the sender's intent:
        {main_request_descriptions}
        Segment:
        ---
        {segment}
        ---
        Provide a brief reasoning and focus on the specific action requested.
        """
        try:
            main_result = classifier(main_prompt, main_request_types)
            top_main_request = main_result["labels"][0]
            main_confidence = main_result["scores"][0]
            main_reason = REQUEST_TYPES[top_main_request]["description"]

            sub_request_types = REQUEST_TYPES[top_main_request].get("subcategories", {})
            if sub_request_types:
                sub_request_labels = list(sub_request_types.keys())
                sub_request_descriptions = "\n".join([f"- {key}: {value}" for key, value in sub_request_types.items()])
                sub_prompt = f"""
                Classify this segment into a subcategory of '{top_main_request}' based on the sender's intent:
                {sub_request_descriptions}
                Segment:
                ---
                {segment}
                ---
                Provide a brief reasoning.
                """
                sub_result = classifier(sub_prompt, sub_request_labels)
                top_sub_request = sub_result["labels"][0]
                sub_confidence = sub_result["scores"][0]
                sub_reason = sub_request_types[top_sub_request]
                confidence = round(0.7 * main_confidence + 0.3 * sub_confidence, 4)
            else:
                top_sub_request = "NA"
                sub_reason = main_reason
                confidence = round(main_confidence, 4)

            # Extract fields from this segment only
            extracted_data = extract_fields(segment, top_main_request)
            
            # If fields are missing, try full content as fallback
            for field, value in extracted_data.items():
                if value == "Not Found":
                    full_match = re.search(PATTERNS[field], content, re.IGNORECASE)
                    if full_match:
                        extracted_data[field] = full_match.group(1).strip()

            results.append({
                "request_type": top_main_request,
                "sub_request_type": top_sub_request,
                "reason": sub_reason,
                "confidence": confidence,
                "extracted_data": extracted_data,
                "is_primary": False
            })
        except Exception as e:
            logger.error(f"Error classifying segment: {e}")
            results.append({
                "request_type": "NA",
                "sub_request_type": "NA",
                "reason": f"Classification error: {str(e)}",
                "confidence": 0.0,
                "extracted_data": {},
                "is_primary": False
            })

    # Deduplicate identical requests (same type and subtype)
    unique_results = []
    seen = set()
    for result in results:
        key = (result["request_type"], result["sub_request_type"])
        if key not in seen and result["request_type"] != "NA":
            unique_results.append(result)
            seen.add(key)

    # Determine primary intent (highest confidence)
    if unique_results:
        primary_idx = max(range(len(unique_results)), key=lambda i: unique_results[i]["confidence"])
        unique_results[primary_idx]["is_primary"] = True
        logger.info(f"Primary intent: {unique_results[primary_idx]['request_type']} - {unique_results[primary_idx]['sub_request_type']}")
    else:
        unique_results = [{"request_type": "NA", "sub_request_type": "NA", "reason": "No valid requests found.", "confidence": 0.0, "extracted_data": {}, "is_primary": True}]

    return unique_results

# --- Main Processing Logic ---

def process_email_directory(directory: str) -> List[Dict]:
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

        classifications = classify_email(text)
        results.append({
            "file": filename,
            "classifications": classifications
        })

    return results

# --- Execution ---

if __name__ == "__main__":
    EMAIL_DIRECTORY = "/Users/paramita.santra/impks/hackhive-2025/emails-new"
    OUTPUT_FILE = "classification_results.json"

    classification_results = process_email_directory(EMAIL_DIRECTORY)

    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(classification_results, f, indent=2)
        logger.info(f"Classification completed! Results saved in {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results to {OUTPUT_FILE}: {e}")

    print("\n📌 Classification Results:\n")
    print(json.dumps(classification_results, indent=2))