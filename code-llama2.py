import os
import json
import pdfplumber
import mailparser
from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Request Type Definitions
request_types = {
    "Money Movement - Inbound": "Any money coming into the bank, such as customer loan repayments, incoming wire transfers, and deposits.",
    "Money Movement - Outbound": "Any money going out of the bank, such as loan disbursements, refunds, or wire transfers sent to customers.",
    "Billing Issue": "Customer inquiries related to incorrect charges, missing payments, or overcharges on accounts.",
    "Technical Support": "Requests related to system access issues, account login problems, or software glitches.",
    "Account Management": "Requests for account modifications, user role changes, or updates to customer details.",
    "General Inquiry": "Any question or request that does not fit into other categories.",
    "Contract Renewal Request": "Emails related to renewing contracts, updating agreements, or extending financial services."
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

# Function to classify email content
def classify_email(content):
    if not content.strip():
        return {"request_type": "NA", "reason": "No meaningful content found.", "confidence": 0.0}

    # Construct classification prompt
    request_type_descriptions = "\n".join([f"- {key}: {value}" for key, value in request_types.items()])
    classification_prompt = f"""
    You are an AI email classifier for a Loan Services bank. Your job is to classify emails into predefined request types based on their content.

    Here are the request types and their meanings:
    {request_type_descriptions}

    Given the following email:
    ---
    {content}
    ---

    Classify the email into one of the request types listed above and provide a brief reasoning.
    """

    try:
        result = classifier(classification_prompt, list(request_types.keys()))

        top_class = result["labels"][0]
        confidence = result["scores"][0]
        reason = request_types[top_class]

        return {
            "request_type": top_class,
            "reason": reason,
            "confidence": round(confidence, 4)  # Round to 4 decimal places
        }
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return {"request_type": "NA", "reason": "Classification error.", "confidence": 0.0}

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
