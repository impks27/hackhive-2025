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
    "Adjustment": {
    "description": "Revisions or modifications made to existing financial agreements, obligations, or fee structures. Adjustments may involve correcting transaction details, restructuring financial terms, or redistributing funds within a deal. These changes are typically driven by contractual amendments, compliance requirements, or business needs.",
    "fields": ["deal_name", "amount", "transaction_date"]
},
"AU Transfer": {
    "description": "Fund transfers related to Allocation Units (AU), where a principal amount is moved between different financial structures, accounts, or investment allocations. AU transfers typically occur in structured finance agreements and may involve capital reallocation without changing the total committed amount.",
    "fields": ["deal_name", "amount", "transaction_date"]
}
,
    "Closing Notice": {
    "description": "Notifications or actions related to terminating or modifying an existing financial agreement, loan position, or investment structure. This may involve the reallocation of funds, changes to principal amounts, or associated fees. Closing notices typically indicate the finalization or adjustment of financial obligations within a deal.",
    "subcategories": {
        "Reallocation Fees": "Charges incurred when reallocating funds, assets, or positions within an agreement, typically due to structural changes in the deal.",
        "Amendment Fees": "Fees applied for modifications or contractual adjustments to the terms of an agreement, including extensions, rate changes, or restructuring.",
        "Reallocation Principal": "An adjustment to the principal amount during a reallocation process, often related to debt restructuring or capital reallocation within a financial agreement."
    },
    "fields": ["deal_name", "transaction_date", "amount"]
},
    "Commitment Change": {
    "description": "Adjustments to the level of committed financial resources or obligations within a loan, credit facility, or investment agreement. These changes may result from amendments, restructurings, or re-evaluations of funding needs and can impact borrowing capacity, liquidity, or financial commitments. Commitment changes can involve increases, decreases, or the rolling over of existing positions without additional cash settlement.",
    "subcategories": {
        "Decrease": "A reduction in the committed amount or financial obligation, often due to partial repayments, reduced funding needs, facility downsizing, or contractual amendments.",
        "Increase": "An increase in the committed amount or financial obligation, typically resulting from additional funding requests, credit line extensions, or deal expansions.",
        "Cashless Roll": "A transaction where an existing commitment is rolled over or extended without requiring a new cash settlement, often used to restructure obligations or maintain continuity in funding."
    },
    "fields": ["deal_name", "amount", "transaction_date"]
},
    "Fee Payment": {
    "description": "Payments related to fees associated with financial agreements, loan services, or credit facilities. These payments cover recurring or transaction-based charges incurred as part of maintaining financial obligations.",
    "subcategories": {
        "Ongoing Fee": "Recurring fees charged for continuous services, loan maintenance, or administrative costs.",
        "Letter of Credit Fee": "Fees associated with issuing, maintaining, or amending a letter of credit, typically charged to facilitate trade or financial transactions."
    },
    "fields": ["deal_name", "amount", "transaction_date", "account_number"]
},

    "Money Movement - Inbound": {
    "description": "Transactions involving funds being received by the bank, such as loan repayments, interest payments, or capital contributions. These transactions may include customer repayments, scheduled interest settlements, and combined payments covering multiple financial obligations.",
    "subcategories": {
        "Principal": "A payment that solely covers the repayment of the original loan amount without any interest or additional charges.",
        "Interest": "A payment made to cover accrued interest on a loan or financial obligation, separate from the principal amount.",
        "Principal + Interest": "A combined payment that includes both the repayment of the original loan amount and the interest accrued on it.",
        "Principal + Interest + Fee": "A comprehensive payment that includes the principal repayment, interest charges, and any associated service or processing fees."
    },
    "fields": ["deal_name", "amount", "transaction_date", "account_number"]
}
,
   "Money Movement - Outbound": {
    "description": "Transactions involving funds leaving the bank, such as loan disbursements, scheduled payouts, or international transfers. These transactions may include time-sensitive disbursements or foreign currency transactions requiring special handling and exchange rate considerations.",
    "subcategories": {
        "Timebound": "A scheduled or deadline-driven fund transfer, such as a loan disbursement, contractually obligated payout, or settlement that must be completed within a specific timeframe.",
        "Foreign Currency": "An outbound transaction where funds are sent in a currency different from the base account currency, requiring exchange rate conversions and potentially additional processing fees."
    },
    "fields": ["deal_name", "amount", "transaction_date", "currency"]
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
                "confidence": round((0.7 * main_confidence) + (0.3 * sub_confidence), 4),
                #"confidence": round(sub_confidence, 4),
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
    EMAIL_DIRECTORY = "/Users/paramita.santra/impks/hackhive-2025/emails-new"
    
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