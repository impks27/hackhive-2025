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

# Define request types and subcategories with examples
REQUEST_TYPES = {
    "Adjustment": {
        "description": "Revisions or modifications made to existing financial agreements, obligations, or fee structures.",
        "fields": ["deal_name", "amount", "transaction_date"],
        "examples": [
            "Please adjust the fee structure for Deal XYZ to $5000 effective 03/15/2025.",
            "Modify the principal of Deal RST by $7500 on 03/25/2025."
        ]
    },
    "AU Transfer": {
        "description": "Fund transfers related to Allocation Units (AU), where a principal amount is moved between different financial structures.",
        "fields": ["deal_name", "amount", "transaction_date"],
        "examples": [
            "Transfer $10,000 from Deal ABC to Deal DEF on 03/20/2025.",
            "Move $8,500 in Allocation Units for Deal UVW to Deal XYZ on 03/23/2025."
        ]
    },
    "Closing Notice": {
        "description": "Notifications or actions related to terminating or modifying an existing financial agreement.",
        "subcategories": {
            "Reallocation Fees": {
                "description": "Charges incurred when reallocating funds, assets, or positions within an agreement.",
                "examples": [
                    "Charge $200 in reallocation fees for Deal GHI on 03/22/2025.",
                    "Apply a $350 reallocation fee to Deal PQR on 03/27/2025."
                ]
            },
            "Amendment Fees": {
                "description": "Fees applied for modifications or contractual adjustments to the terms of an agreement.",
                "examples": [
                    "Apply a $150 amendment fee to Deal JKL effective 03/18/2025.",
                    "Charge $275 amendment fee for Deal MNO on 03/24/2025."
                ]
            },
            "Reallocation Principal": {
                "description": "An adjustment to the principal amount during a reallocation process.",
                "examples": [
                    "Reallocate $25,000 principal for Deal MNO on 03/21/2025.",
                    "Adjust principal by $15,000 for Deal STU during reallocation on 03/26/2025."
                ]
            }
        },
        "fields": ["deal_name", "transaction_date", "amount"],
        "examples": [
            "Issue a closing notice for Deal PQR with a $1000 adjustment on 03/19/2025.",
            "Notify termination of Deal BCD with $2,500 adjustment on 03/28/2025."
        ]
    },
    "Commitment Change": {
        "description": "Adjustments to the level of committed financial resources or obligations within a loan or credit facility.",
        "subcategories": {
            "Decrease": {
                "description": "A reduction in the committed amount or financial obligation.",
                "examples": [
                    "Decrease commitment for Deal STU by $5,000 on 03/17/2025.",
                    "Reduce Deal EFG commitment by $3,200 on 03/29/2025."
                ]
            },
            "Increase": {
                "description": "An increase in the committed amount or financial obligation.",
                "examples": [
                    "Increase commitment for Deal VWX by $7,500 on 03/16/2025.",
                    "Add $9,000 to the commitment for Deal HIJ on 03/30/2025."
                ]
            },
            "Cashless Roll": {
                "description": "A transaction where an existing commitment is rolled over without cash settlement.",
                "examples": [
                    "Perform a cashless roll for Deal YZA on 03/20/2025.",
                    "Roll over commitment for Deal KLM without cash on 03/31/2025."
                ]
            }
        },
        "fields": ["deal_name", "amount", "transaction_date"],
        "examples": [
            "Adjust commitment for Deal BCD by $3,000 on 03/14/2025.",
            "Change commitment level for Deal NOP by $4,500 on 03/27/2025."
        ]
    },
    "Fee Payment": {
        "description": "Payments related to fees associated with financial agreements or loan services.",
        "subcategories": {
            "Ongoing Fee": {
                "description": "Recurring fees charged for continuous services or loan maintenance.",
                "examples": [
                    "Process an ongoing fee of $300 for Deal EFG on 03/22/2025, account 12345.",
                    "Charge $450 ongoing fee for Deal QRS on 03/28/2025, account 99887."
                ]
            },
            "Letter of Credit Fee": {
                "description": "Fees associated with issuing or amending a letter of credit.",
                "examples": [
                    "Charge a $400 letter of credit fee for Deal HIJ on 03/19/2025, account 67890.",
                    "Apply $325 fee for letter of credit on Deal TUV on 03/26/2025, account 44556."
                ]
            }
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"],
        "examples": [
            "Submit a fee payment of $250 for Deal KLM on 03/15/2025, account 54321.",
            "Process $175 fee payment for Deal WXY on 03/24/2025, account 77665."
        ]
    },
    "Money Movement - Inbound": {
        "description": "Transactions involving funds being received by the bank, such as loan repayments or interest payments.",
        "subcategories": {
            "Principal": {
                "description": "Payment covering only the original loan amount.",
                "examples": [
                    "Receive $15,000 principal payment for Deal NOP on 03/18/2025, account 98765.",
                    "Record $12,500 principal for Deal ZAB on 03/27/2025, account 22334."
                ]
            },
            "Interest": {
                "description": "Payment covering accrued interest on a loan.",
                "examples": [
                    "Record $1,200 interest payment for Deal QRS on 03/20/2025, account 45678.",
                    "Receive $950 interest for Deal CDE on 03/29/2025, account 66778."
                ]
            },
            "Principal + Interest": {
                "description": "Combined payment of principal and interest.",
                "examples": [
                    "Process $20,000 principal + interest payment for Deal TUV on 03/21/2025, account 11223.",
                    "Receive $18,000 principal and interest for Deal FGH on 03/30/2025, account 88990."
                ]
            },
            "Principal + Interest + Fee": {
                "description": "Payment including principal, interest, and fees.",
                "examples": [
                    "Receive $25,000 including principal, interest, and fees for Deal WXY on 03/22/2025, account 33445.",
                    "Process $30,000 payment covering principal, interest, and fees for Deal IJK on 03/31/2025, account 55667."
                ]
            }
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"],
        "examples": [
            "Record an inbound payment of $10,000 for Deal ZAB on 03/17/2025, account 66778.",
            "Receive $14,000 inbound for Deal MNO on 03/25/2025, account 77889."
        ]
    },
    "Money Movement - Outbound": {
        "description": "Transactions involving funds leaving the bank, such as loan disbursements or transfers.",
        "subcategories": {
            "Timebound": {
                "description": "Scheduled or deadline-driven fund transfer.",
                "examples": [
                    "Disburse $30,000 for Deal CDE on 03/15/2025, timebound.",
                    "Send $22,000 for Deal RST on 03/28/2025, timebound."
                ]
            },
            "Foreign Currency": {
                "description": "Outbound transaction in a different currency.",
                "examples": [
                    "Transfer €25,000 for Deal FGH on 03/16/2025 in foreign currency.",
                    "Disburse £18,000 for Deal UVW on 03/29/2025 in foreign currency."
                ]
            }
        },
        "fields": ["deal_name", "amount", "transaction_date", "currency"],
        "examples": [
            "Send $12,000 outbound for Deal IJK on 03/14/2025 in USD.",
            "Transfer $16,500 for Deal YZA on 03/26/2025 in USD."
        ]
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

# Function to classify email content with examples
def classify_email(content):
    if not content.strip():
        return {"request_type": "NA", "sub_request_type": "NA", "confidence": 0.0, "extracted_data": {}}

    # Construct prompt for main request types with examples
    main_prompt = "Classify the following financial request into one of these categories:\n"
    for req_type, details in REQUEST_TYPES.items():
        main_prompt += f"{req_type} ({details['description']}):\n"
        for example in details["examples"]:
            main_prompt += f"  - Example: {example}\n"
    main_prompt += f"\nText to classify:\n{content}"

    # Classify main request type
    main_result = classifier(main_prompt, list(REQUEST_TYPES.keys()), multi_label=False)
    top_request_type = main_result["labels"][0]
    confidence = main_result["scores"][0]

    # Handle subcategories if they exist
    subcategories = REQUEST_TYPES.get(top_request_type, {}).get("subcategories", {})
    if subcategories:
        # Construct prompt for subcategories with examples
        sub_prompt = f"Given the request type '{top_request_type}', classify the following into a subcategory:\n"
        for subcat, details in subcategories.items():
            sub_prompt += f"{subcat} ({details['description']}):\n"
            for example in details["examples"]:
                sub_prompt += f"  - Example: {example}\n"
        sub_prompt += f"\nText to classify:\n{content}"
        
        sub_result = classifier(sub_prompt, list(subcategories.keys()), multi_label=False)
        top_subcategory = sub_result["labels"][0]
    else:
        top_subcategory = "NA"

    # Extract data using regex
    extracted_data = {
        field: re.search(PATTERNS[field], content).group(1) if re.search(PATTERNS[field], content) else "Not Found"
        for field in REQUEST_TYPES.get(top_request_type, {}).get("fields", [])
    }

    return {
        "request_type": top_request_type,
        "sub_request_type": top_subcategory,
        "confidence": confidence,
        "extracted_data": extracted_data
    }

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
    folder_path = "emails-accuracy-test/"  # Update with the actual path
    results = process_email_folder(folder_path)
    print(json.dumps(results, indent=2))