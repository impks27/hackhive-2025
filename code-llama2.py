import os
import json
import re
import pdfplumber
import mailparser
from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Request Type Definitions (Main & Subcategories)
request_types = {
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
patterns = {
    "deal_name": r"Deal Name[:\s]*([\w\s-]+)",
    "amount": r"Amount[:\s]*\$?([\d,]+\.?\d*)",
    "transaction_date": r"Transaction Date[:\s]*(\d{2}/\d{2}/\d{4})",
    "invoice_number": r"Invoice Number[:\s]*(\w+)",
    "billing_date": r"Billing Date[:\s]*(\d{2}/\d{2}/\d{4})"
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
        return ""

# Function to extract text from .eml emails
def extract_text_from_eml(eml_path):
    try:
        mail = mailparser.parse_from_file(eml_path)
        return f"Subject: {mail.subject}\n{mail.body}"
    except Exception as e:
        print(f"❌ Error extracting text from {eml_path}: {e}")
        return ""

# Function to extract specific fields based on request type
def extract_fields(content, request_type):
    extracted_data = {}
    fields = request_types.get(request_type, {}).get("fields", [])

    for field in fields:
        pattern = patterns.get(field)
        if pattern:
            match = re.search(pattern, content, re.IGNORECASE)
            extracted_data[field] = match.group(1) if match else "Not Found"

    return extracted_data

# Function to classify email content into main and subcategories
def classify_email(content):
    if not content.strip():
        return {
            "request_type": "NA",
            "sub_request_type": "NA",
            "reason": "No meaningful content found.",
            "confidence": 0.0,
            "extracted_data": {}
        }

    # Classify main request type
    main_request_types = list(request_types.keys())
    main_request_descriptions = "\n".join([f"- {key}: {value['description']}" for key, value in request_types.items()])

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
        main_result = classifier(main_prompt, main_request_types)
        top_main_request = main_result["labels"][0]
        main_confidence = main_result["scores"][0]
        main_reason = request_types[top_main_request]["description"]

        # Classify subcategory if applicable
        sub_request_types = request_types[top_main_request].get("subcategories", {})
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

            # Extract relevant fields
            extracted_data = extract_fields(content, top_main_request)

            return {
                "request_type": top_main_request,
                "sub_request_type": top_sub_request,
                "reason": sub_reason,
                "confidence": round(min(main_confidence, sub_confidence), 4),
                "extracted_data": extracted_data
            }

        # If no subcategories exist, return only the main request type
        extracted_data = extract_fields(content, top_main_request)
        return {
            "request_type": top_main_request,
            "sub_request_type": "NA",
            "reason": main_reason,
            "confidence": round(main_confidence, 4),
            "extracted_data": extracted_data
        }

    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return {
            "request_type": "NA",
            "sub_request_type": "NA",
            "reason": "Classification error.",
            "confidence": 0.0,
            "extracted_data": {}
        }

# Process all emails in a directory
def process_email_directory(directory):
    results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".eml"):
            text = extract_text_from_eml(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        classification = classify_email(text)
        results.append({"file": filename, **classification})

    return results

# 🔹 Update the directory path to where your PDFs and EMLs are stored
email_directory = "/Users/paramita.santra/impks/hackhive-2025/emails"
classification_results = process_email_directory(email_directory)

# Save results as JSON
output_file = "classification_results.json"
with open(output_file, "w") as f:
    json.dump(classification_results, f, indent=2)

print(f"✅ Classification completed! Results saved in {output_file}")

# 🔹 Print classification results
print("\n📌 Classification Results:\n")
print(json.dumps(classification_results, indent=2))
