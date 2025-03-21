import os
import json
import pdfplumber
import mailparser
from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define request types with descriptions
REQUEST_TYPES = {
    "Billing Issue": "Emails related to invoice discrepancies, payment failures, or refund requests.",
    "Technical Support": "Emails requesting help with software issues, login problems, or troubleshooting.",
    "Account Management": "Emails regarding account changes, password resets, or user access updates.",
    "General Inquiry": "Emails asking about company services, policies, or general information.",
    "Contract Renewal Request": "Emails discussing contract renewal, extensions, or new agreements."
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text.strip() if text else None
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return None

# Function to extract text from .eml files
def extract_text_from_eml(eml_path):
    try:
        parsed_mail = mailparser.parse_from_file(eml_path)
        text = parsed_mail.subject + "\n" + parsed_mail.body
        return text.strip() if text else None
    except Exception as e:
        print(f"❌ Error extracting EML text: {e}")
        return None

# Classify email content
def classify_email(text):
    if not text:
        return {"request_type": "NA", "reason": "No text found"}

    result = classifier(text, list(REQUEST_TYPES.keys()), multi_label=False)
    classification = result["labels"][0]
    confidence = result["scores"][0]

    return {
        "request_type": classification,
        "confidence": round(confidence, 2),
        "reason": f"{REQUEST_TYPES[classification]} (Confidence: {round(confidence * 100, 2)}%)"
    }

# Process all files in a directory
def process_emails(directory):
    results = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_extension = filename.split('.')[-1].lower()

        if file_extension == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_extension == "eml":
            text = extract_text_from_eml(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue
        
        classification = classify_email(text)
        classification["filename"] = filename  # Add filename to result
        results.append(classification)

    return results

# Directory containing emails
EMAILS_DIR = "/Users/paramita.santra/impks/hackhive-2025/emails"

# Run classification
classification_results = process_emails(EMAILS_DIR)

# Print results
print(json.dumps(classification_results, indent=2))
