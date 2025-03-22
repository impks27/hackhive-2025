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
    #     "description": "This is when we change something about a money deal that’s already set up. It could be tweaking how much someone owes, updating fees, or fixing details in an agreement. It’s like making small updates to keep things right.",
    #     "fields": ["deal_name", "amount", "transaction_date"],
    #     "examples": [
    #         "Please adjust the fee structure for Deal XYZ to $5000 effective 03/15/2025.",
    #         "Modify the principal of Deal RST by $7500 on 03/25/2025."
    #     ]
    # },
    "AU Transfer": {
        "description": "This is about moving money between different parts of a financial setup, called Allocation Units (AU). It’s like shifting a chunk of cash from one bucket to another in the same system, usually the main amount someone borrowed.",
        "fields": ["deal_name", "amount", "transaction_date"],
        "examples": [
            "Transfer $10,000 from Deal ABC to Deal DEF on 03/20/2025.",
            "Move $8,500 in Allocation Units for Deal UVW to Deal XYZ on 03/23/2025."
        ]
    },
    "Closing Notice": {
        "description": "This is when we tell people a money deal is ending or changing in a big way. It’s like sending a heads-up that we’re wrapping things up or tweaking something major, so everyone knows what’s happening.",
        "subcategories": {
            "Reallocation Fees": {
                "description": "This is extra money we charge when we move funds or assets around inside a deal. Think of it as a fee for shuffling things to new spots."
            },
            "Amendment Fees": {
                "description": "This is a charge for changing the rules or terms of a deal. It’s like a fee for editing the agreement to make it work better."
            },
            "Reallocation Principal": {
                "description": "This is when we adjust the main amount of money in a deal while moving it around. It’s like updating the core cash amount during a shuffle."
            }
        },
        "fields": ["deal_name", "transaction_date", "amount"],
        "examples": [
            "Issue a closing notice for Deal PQR with a $1000 adjustment on 03/19/2025.",
            "Notify termination of Deal BCD with $2,500 adjustment on 03/28/2025."
        ]
    },
    "Commitment Change": {
        "description": "This is when we adjust how much money we’ve promised to give or hold for a loan or credit deal. It’s like changing our pledge—maybe giving more, less, or just rolling it over without extra cash.",
        "subcategories": {
            "Decrease": {
                "description": "This is when we lower the amount of money we promised. It’s like saying we’ll commit less cash than before."
            },
            "Increase": {
                "description": "This is when we raise the amount of money we promised. It’s like agreeing to put more cash on the table."
            },
            "Cashless Roll": {
                "description": "This is when we keep the promise going without adding new money. It’s like renewing the deal as-is, no cash needed."
            }
        },
        "fields": ["deal_name", "amount", "transaction_date"],
        "examples": [
            "Adjust commitment for Deal BCD by $3,000 on 03/14/2025.",
            "Change commitment level for Deal NOP by $4,500 on 03/27/2025."
        ]
    },
    "Fee Payment": {
        "description": "This is when we handle payments for extra charges tied to money deals or loans. It’s like collecting the costs for keeping things running or making special changes.",
        "subcategories": {
            "Ongoing Fee": {
                "description": "This is a regular charge we keep taking for ongoing services, like maintaining a loan. It’s like a monthly bill for keeping everything in order."
            },
            "Letter of Credit Fee": {
                "description": "This is a charge for setting up or changing a promise to pay someone later. It’s like a fee for a financial IOU we issue or tweak."
            }
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"],
        "examples": [
            "Submit a fee payment of $250 for Deal KLM on 03/15/2025, account 54321.",
            "Process $175 fee payment for Deal WXY on 03/24/2025, account 77665."
        ]
    },
    "Money Movement - Inbound": {
        "description": "This is when money comes into the bank from outside. It could be someone paying back a loan, sending interest, or covering fees—basically any cash flowing our way.",
        "subcategories": {
            "Principal": {
                "description": "This is when someone pays back just the main amount they borrowed. It’s the core money, no extras like interest."
            },
            "Interest": {
                "description": "This is when someone pays the extra cost for borrowing money. It’s like the fee they owe on top of the loan."
            },
            "Principal + Interest": {
                "description": "This is when someone pays back both the main amount and the extra cost together. It’s a combo payment."
            },
            "Principal + Interest + Fee": {
                "description": "This is when someone sends money that covers everything—the main amount, the extra cost, and any fees. It’s like paying the full bill at once."
            }
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"],
        "examples": [
            "Record an inbound payment of $10,000 for Deal ZAB on 03/17/2025, account 66778.",
            "Receive $14,000 inbound for Deal MNO on 03/25/2025, account 77889."
        ]
    },
    "Money Movement - Outbound": {
        "description": "This is when money leaves the bank to go somewhere else. It includes things like sending out loans, paying someone, or moving funds to another account or place. Basically, any time the bank sends cash outward.",
        "subcategories": {
            "Timebound": {
                "description": "This is when money has to be sent out by a specific time or date. Think of it like a deadline—maybe a payment is due, or funds need to reach someone by a set day. It’s planned and time-sensitive."
            },
            "Foreign Currency": {
                "description": "This is when the bank sends money in a currency that’s not the usual one (like dollars). For example, sending euros or pounds instead of U.S. dollars to another country or account."
            }
        },
        "fields": ["deal_name", "amount", "transaction_date", "currency"],
        "examples": [
            "Send $12,000 outbound for Deal IJK on 03/14/2025 in USD.",
            "Transfer $16,500 for Deal YZA on 03/26/2025 in USD."
        ]
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
                #confidence = round(0.7 * main_confidence + 0.3 * sub_confidence, 4)
                confidence = round(max(main_confidence, sub_confidence), 4)
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
    EMAIL_DIRECTORY = "/Users/paramita.santra/impks/hackhive-2025/test_pdfs"  #emails-new
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