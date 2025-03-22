import os
import json
import re
import pdfplumber
import mailparser
from transformers import pipeline
import logging
import hashlib
from typing import List, Dict, Tuple

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

# Request Type Definitions
REQUEST_TYPES = {
    "Adjustment": {
        "description": "Revisions or modifications made to existing financial agreements, obligations, or fee structures.",
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "AU Transfer": {
        "description": "Fund transfers related to Allocation Units (AU), moving principal between accounts or allocations.",
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Closing Notice": {
        "description": "Notifications or actions related to terminating or modifying an agreement.",
        "subcategories": {
            "Reallocation Fees": "Charges for reallocating funds or assets.",
            "Amendment Fees": "Fees for contractual adjustments.",
            "Reallocation Principal": "Adjustment to principal during reallocation."
        },
        "fields": ["deal_name", "transaction_date", "amount"]
    },
    "Commitment Change": {
        "description": "Adjustments to committed financial resources.",
        "subcategories": {
            "Decrease": "Reduction in committed amount.",
            "Increase": "Increase in committed amount.",
            "Cashless Roll": "Rollover of commitment without cash settlement."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Fee Payment": {
        "description": "Payments related to fees for financial agreements.",
        "subcategories": {
            "Ongoing Fee": "Recurring fees for services.",
            "Letter of Credit Fee": "Fees for letter of credit services."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Money Movement - Inbound": {
        "description": "Funds received by the bank, e.g., loan repayments.",
        "subcategories": {
            "Principal": "Payment of loan principal.",
            "Interest": "Payment of accrued interest.",
            "Principal + Interest": "Combined principal and interest payment.",
            "Principal + Interest + Fee": "Payment covering all three."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Money Movement - Outbound": {
        "description": "Funds leaving the bank, e.g., disbursements.",
        "subcategories": {
            "Timebound": "Scheduled or deadline-driven transfer.",
            "Foreign Currency": "Transfer in a foreign currency."
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
    "currency": r"Currency[:\s]*([A-Z]{3})",
    "expiration_date": r"Expiration Date[:\s]*(\d{2}/\d{2}/\d{4})"
}

# Skill-based routing map
ROUTING_MAP = {
    "Adjustment": "Adjustments Team",
    "AU Transfer": "Transfers Team",
    "Closing Notice": "Closures Team",
    "Commitment Change": "Commitments Team",
    "Fee Payment": "Fees Team",
    "Money Movement - Inbound": "Inbound Payments Team",
    "Money Movement - Outbound": "Outbound Payments Team"
}

# --- Utility Functions ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            logger.info(f"Extracted text from PDF: {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_eml(eml_path: str) -> Tuple[str, str]:
    """Extract text and subject from an .eml email file."""
    try:
        mail = mailparser.parse_from_file(eml_path)
        text = mail.body
        subject = mail.subject
        logger.info(f"Extracted text from EML: {eml_path}")
        return text, subject
    except Exception as e:
        logger.error(f"Error extracting text from {eml_path}: {e}")
        return "", ""

def extract_fields(content: str, request_type: str, attachment_content: str = "") -> Dict[str, str]:
    """Extract fields from email body (priority) and attachments."""
    extracted_data = {}
    fields = REQUEST_TYPES.get(request_type, {}).get("fields", [])

    for field in fields:
        pattern = PATTERNS.get(field)
        if pattern:
            # Prioritize email body for non-numeric fields, attachments for numeric
            if field in ["amount", "transaction_date", "expiration_date"] and attachment_content:
                match = re.search(pattern, attachment_content, re.IGNORECASE)
                extracted_data[field] = match.group(1) if match else "Not Found"
            else:
                match = re.search(pattern, content, re.IGNORECASE)
                extracted_data[field] = match.group(1) if match else "Not Found"
    
    return extracted_data

def compute_content_hash(content: str, subject: str) -> str:
    """Compute a hash for duplicate detection."""
    return hashlib.md5((subject + content).encode()).hexdigest()

# --- Classification Logic ---

def classify_email(email_content: str, attachment_content: str = "") -> List[Dict]:
    """Classify email into multiple request types, detect primary intent."""
    if not email_content.strip():
        logger.warning("Empty email content provided.")
        return [{"request_type": "NA", "sub_request_type": "NA", "reason": "No content", "confidence": 0.0, "extracted_data": {}, "is_primary": False}]

    # Split email into segments (e.g., by paragraphs or sentences) for multi-request detection
    segments = [seg.strip() for seg in re.split(r'\n{2,}|\.\s+', email_content) if seg.strip()]
    results = []

    main_request_types = list(REQUEST_TYPES.keys())
    main_descriptions = "\n".join([f"- {key}: {value['description']}" for key, value in REQUEST_TYPES.items()])

    # Classify each segment
    for i, segment in enumerate(segments):
        main_prompt = f"""
        Classify this email segment into a request type:
        {main_descriptions}
        Segment:
        ---
        {segment}
        ---
        Provide reasoning.
        """
        try:
            main_result = classifier(main_prompt, main_request_types)
            top_main_request = main_result["labels"][0]
            main_confidence = main_result["scores"][0]
            main_reason = REQUEST_TYPES[top_main_request]["description"]

            sub_request_types = REQUEST_TYPES[top_main_request].get("subcategories", {})
            if sub_request_types:
                sub_labels = list(sub_request_types.keys())
                sub_descriptions = "\n".join([f"- {key}: {value}" for key, value in sub_request_types.items()])
                sub_prompt = f"""
                Classify this segment into a subcategory of '{top_main_request}':
                {sub_descriptions}
                Segment:
                ---
                {segment}
                ---
                """
                sub_result = classifier(sub_prompt, sub_labels)
                top_sub_request = sub_result["labels"][0]
                sub_confidence = sub_result["scores"][0]
                sub_reason = sub_request_types[top_sub_request]
                confidence = round((0.7 * main_confidence) + (0.3 * sub_confidence), 4)
            else:
                top_sub_request = "NA"
                sub_reason = main_reason
                confidence = round(main_confidence, 4)

            extracted_data = extract_fields(segment, top_main_request, attachment_content)
            results.append({
                "request_type": top_main_request,
                "sub_request_type": top_sub_request,
                "reason": sub_reason,
                "confidence": confidence,
                "extracted_data": extracted_data,
                "is_primary": False  # Placeholder, updated later
            })
        except Exception as e:
            logger.error(f"Error classifying segment {i}: {e}")
            results.append({"request_type": "NA", "sub_request_type": "NA", "reason": str(e), "confidence": 0.0, "extracted_data": {}, "is_primary": False})

    # Determine primary intent (highest confidence or first valid request)
    if results:
        primary_idx = max(range(len(results)), key=lambda i: results[i]["confidence"] if results[i]["request_type"] != "NA" else -1)
        results[primary_idx]["is_primary"] = True
        logger.info(f"Primary intent: {results[primary_idx]['request_type']} - {results[primary_idx]['sub_request_type']}")

    return results

# --- Processing Logic ---

def process_email_directory(directory: str) -> Dict[str, List[Dict]]:
    """Process emails and attachments, detect duplicates, and route requests."""
    results = {}
    seen_hashes = set()

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        attachment_content = ""

        # Handle PDFs (attachments)
        if filename.lower().endswith(".pdf"):
            attachment_content = extract_text_from_pdf(file_path)
            continue  # Skip standalone PDFs, process with email

        # Handle EMLs (emails)
        elif filename.lower().endswith(".eml"):
            email_content, subject = extract_text_from_eml(file_path)
            content_hash = compute_content_hash(email_content, subject)

            # Duplicate detection
            if content_hash in seen_hashes:
                logger.warning(f"Duplicate email detected: {filename}")
                results[filename] = [{"request_type": "Duplicate", "sub_request_type": "NA", "reason": "Duplicate email", "confidence": 1.0, "extracted_data": {}, "is_primary": True}]
                continue
            seen_hashes.add(content_hash)

            # Classify and extract
            classifications = classify_email(email_content, attachment_content)
            for classification in classifications:
                if classification["request_type"] != "NA":
                    classification["assigned_team"] = ROUTING_MAP.get(classification["request_type"], "Unassigned")
            results[filename] = classifications
        else:
            logger.warning(f"Skipping unsupported file: {filename}")

    return results

# --- Execution ---

if __name__ == "__main__":
    EMAIL_DIRECTORY = "/Users/paramita.santra/impks/hackhive-2025/emails-new"
    OUTPUT_FILE = "classification_results.json"

    # Process emails
    classification_results = process_email_directory(EMAIL_DIRECTORY)

    # Save results
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(classification_results, f, indent=2)
        logger.info(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    # Print results
    print("\n📌 Classification Results:\n")
    print(json.dumps(classification_results, indent=2))