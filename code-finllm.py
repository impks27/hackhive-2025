import os
import json
import re
import pdfplumber
import mailparser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load FinLLaMA Model
MODEL_NAME = "bavest/fin-llama-33b-merged"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)

    # Fix: Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        offload_folder="./offload_weights",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    logger.info("Successfully loaded FinLLaMA model.")
except Exception as e:
    logger.error(f"Failed to load FinLLaMA model: {e}")
    raise

# Request Type Definitions
REQUEST_TYPES = {
    "Adjustment": {"description": "Revisions or modifications to financial agreements.", "fields": ["deal_name", "amount", "transaction_date"]},
    "AU Transfer": {"description": "Fund transfers related to Allocation Units (AU).", "fields": ["deal_name", "amount", "transaction_date"]},
    "Closing Notice": {
        "description": "Notifications related to financial agreement changes.",
        "subcategories": {"Reallocation Fees": "", "Amendment Fees": "", "Reallocation Principal": ""},
        "fields": ["deal_name", "transaction_date", "amount"],
    },
    "Money Movement - Inbound": {
        "description": "Funds received by the bank.",
        "subcategories": {"Principal": "", "Interest": "", "Principal + Interest": ""},
        "fields": ["deal_name", "amount", "transaction_date", "account_number"],
    },
    "Money Movement - Outbound": {
        "description": "Funds leaving the bank.",
        "subcategories": {"Timebound": "", "Foreign Currency": ""},
        "fields": ["deal_name", "amount", "transaction_date", "currency"],
    },
}

# Regex patterns for data extraction
PATTERNS = {
    "deal_name": r"Deal Name[:\s]*([\w\s-]+)",
    "amount": r"Amount[:\s]*\$?([\d,]+\.?\d*)",
    "transaction_date": r"Transaction Date[:\s]*(\d{2}/\d{2}/\d{4})",
    "invoice_number": r"Invoice Number[:\s]*(\w+)",
    "billing_date": r"Billing Date[:\s]*(\d{2}/\d{2}/\d{4})",
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
    """Classify email content into main and subcategories using FinLLaMA."""
    if not content.strip():
        logger.warning("Empty content provided for classification.")
        return {"request_type": "NA", "sub_request_type": "NA", "reason": "No content.", "confidence": 0.0, "extracted_data": {}}

    # Prepare main classification prompt
    main_prompt = f"""
    Classify the following email into one of the predefined financial request types.

    Request Types:
    {', '.join(REQUEST_TYPES.keys())}

    Email Content:
    {content}

    Return only the request type and a confidence score between 0 and 1.
    """

    # Generate classification response
    inputs = tokenizer(main_prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to correct device
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)  
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract main request type
    top_main_request = next((req for req in REQUEST_TYPES if req in result_text), "NA")
    main_confidence = 0.9  # Mock confidence

    # Handle subcategories
    sub_request_types = REQUEST_TYPES.get(top_main_request, {}).get("subcategories", {})
    top_sub_request, sub_confidence = "NA", 0.0
    if sub_request_types:
        sub_prompt = f"""
        Now that the email is classified as '{top_main_request}', determine the specific subcategory.

        Subcategories:
        {', '.join(sub_request_types.keys())}

        Email Content:
        {content}
        """
        sub_inputs = tokenizer(sub_prompt, return_tensors="pt", truncation=True, padding=True)
        sub_inputs = {k: v.to(device) for k, v in sub_inputs.items()}  # Move to correct device
        with torch.no_grad():
            sub_outputs = model.generate(**sub_inputs, max_new_tokens=50)  
        sub_result_text = tokenizer.decode(sub_outputs[0], skip_special_tokens=True)

        top_sub_request = next((sub for sub in sub_request_types if sub in sub_result_text), "NA")
        sub_confidence = 0.85  # Mock confidence score

    # Confidence score (Weighted average)
    confidence = round((0.7 * main_confidence) + (0.3 * sub_confidence), 4)

    # Extract relevant fields
    extracted_data = extract_fields(content, top_main_request)

    return {
        "request_type": top_main_request,
        "sub_request_type": top_sub_request,
        "reason": REQUEST_TYPES.get(top_main_request, {}).get("description", "Unknown"),
        "confidence": confidence,
        "extracted_data": extracted_data,
    }

def process_email_directory(directory):
    """Process all emails in a directory and classify them."""
    results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        text = extract_text_from_pdf(file_path) if filename.lower().endswith(".pdf") else extract_text_from_eml(file_path)
        classification = classify_email(text)
        results.append({"file": filename, **classification})

    return results

# --- Execution ---
if __name__ == "__main__":
    EMAIL_DIRECTORY = "/Users/paramita.santra/impks/hackhive-2025/emails-new"
    classification_results = process_email_directory(EMAIL_DIRECTORY)

    OUTPUT_FILE = "classification_results.json"
    with open(OUTPUT_FILE, "w") as f:
        json.dump(classification_results, f, indent=2)

    print(json.dumps(classification_results, indent=2))
