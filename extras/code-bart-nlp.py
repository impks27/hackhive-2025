import os
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pdfplumber
import mailparser
from transformers import pipeline
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy-loaded classifiers
def get_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_sentiment_classifier():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Updated Request Type Definitions
REQUEST_TYPES = {
    "Adjustment": {
        "description": "Modifications to existing financial agreements or fee structures.",
        "subcategories": {
            "Reallocation Fees": "Fees associated with redistributing funds or resources to different accounts or purposes.",
            "Amendment Fees": "Charges for altering the terms or conditions of an existing agreement."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "AU Transfer": {
        "description": "Transfers related to allocation units, typically involving principal amounts.",
        "subcategories": {
            "Reallocation Principal": "Movement of the original investment or principal amount to a different allocation."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Closing Notice": {
        "description": "Notifications or actions related to terminating an agreement or position.",
        "subcategories": {
            "Cashless Roll": "A transaction where an existing position is rolled over without cash settlement."
        },
        "fields": ["deal_name", "transaction_date"]
    },
    "Commitment Change": {
        "description": "Adjustments to the level of committed resources or obligations.",
        "subcategories": {
            "Decrease": "Reduction in the committed amount or obligation.",
            "Increase": "Increase in the committed amount or obligation."
        },
        "fields": ["deal_name", "amount", "transaction_date"]
    },
    "Fee Payment": {
        "description": "Payments related to various types of fees or financial obligations.",
        "subcategories": {
            "Ongoing Fee": "Recurring fees charged for continuous services or maintenance.",
            "Letter of Credit Fee": "Fees for issuing or maintaining a letter of credit.",
            "Principal": "Payment covering only the principal amount owed.",
            "Interest": "Payment covering only the interest accrued.",
            "Principal + Interest": "Combined payment of both principal and interest.",
            "Principal+Interest+Fee": "Comprehensive payment including principal, interest, and additional fees."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Money Movement-Inbound": {
        "description": "Funds or resources being received or deposited.",
        "subcategories": {
            "Timebound": "Inbound transfers restricted by specific time conditions or deadlines."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    },
    "Money Movement-Outbound": {
        "description": "Funds or resources being sent or withdrawn.",
        "subcategories": {
            "Foreign Currency": "Outbound transfers involving conversion to or payment in foreign currency."
        },
        "fields": ["deal_name", "amount", "transaction_date", "account_number"]
    }
}

# Enhanced regex patterns (unchanged, but kept compact)
PATTERNS = {
    "deal_name": [r"Deal Name[:\s]+([\w\s-]{3,50})", r"Deal[:\s]+([\w\s-]{3,50})", r"Loan Name[:\s]+([\w\s-]{3,50})", r"Reference[:\s]+([\w\s-]{3,50})"],
    "amount": [r"Amount[:\s]*\$?\s*([\d,]{1,15}\.?\d{0,2})\b", r"Payment[:\s]*\$?\s*([\d,]{1,15}\.?\d{0,2})\b", r"Total[:\s]*\$?\s*([\d,]{1,15}\.?\d{0,2})\b", r"\$\s*([\d,]{1,15}\.?\d{0,2})\b"],
    "transaction_date": [r"Transaction Date[:\s]*(\d{2}[/-]\d{2}[/-]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})", r"Date[:\s]*(\d{2}[/-]\d{2}[/-]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})", r"Payment Date[:\s]*(\d{2}[/-]\d{2}[/-]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})", r"\b(\d{2}[/-]\d{2}[/-]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b"],
    "account_number": [r"Account Number[:\s]*(\d{6,20})", r"Account[:\s]*#?\s*(\d{6,20})", r"A/C[:\s]*#?\s*(\d{6,20})"]
}

class DocumentProcessor:
    def __init__(self, confidence_threshold: float = 0.75):
        self.zero_shot_classifier = None
        self.sentiment_classifier = None
        self.confidence_threshold = confidence_threshold
        
    def _initialize_classifiers(self):
        if self.zero_shot_classifier is None:
            self.zero_shot_classifier = get_zero_shot_classifier()
        if self.sentiment_classifier is None:
            self.sentiment_classifier = get_sentiment_classifier()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)  # Replaced \n with space
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def extract_text_from_eml(self, eml_path: str) -> Tuple[str, str]:
        try:
            mail = mailparser.parse_from_file(eml_path)
            email_body = f"Subject: {mail.subject} Body: {mail.body}"
            attachment_text = ""
            for attachment in mail.attachments:
                content_type = attachment.get("content_type", "").lower()
                payload = attachment.get("payload", "")
                if "text/plain" in content_type:
                    attachment_text += f" Attachment Text: {payload}"
                elif "application/pdf" in content_type:
                    import base64
                    try:
                        pdf_data = base64.b64decode(payload)
                        with open("temp.pdf", "wb") as f:
                            f.write(pdf_data)
                        attachment_text += f" Attachment PDF: {self.extract_text_from_pdf('temp.pdf')}"
                        os.remove("temp.pdf")
                    except Exception as e:
                        logger.error(f"Error processing PDF attachment in {eml_path}: {e}")
            return email_body, attachment_text
        except Exception as e:
            logger.error(f"Error extracting text from {eml_path}: {e}")
            return "", ""

    def _normalize_date(self, date_str: str) -> str:
        try:
            if re.match(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", date_str, re.IGNORECASE):
                date_obj = datetime.strptime(date_str.replace(",", ""), "%B %d %Y")
                return date_obj.strftime("%m/%d/%Y")
            elif re.match(r"\d{2}[/-]\d{2}[/-]\d{4}", date_str):
                return date_str.replace("-", "/")
            return date_str
        except ValueError as e:
            logger.warning(f"Date normalization failed for '{date_str}': {e}")
            return date_str

    def extract_fields(self, content: str, request_type: str) -> Dict[str, any]:
        extracted_data = {}
        fields = REQUEST_TYPES.get(request_type, {}).get("fields", [])
        
        for field in fields:
            field_patterns = PATTERNS.get(field, [])
            value = None
            for pattern in field_patterns:
                try:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        break
                except Exception as e:
                    logger.error(f"Regex error for field '{field}' with pattern '{pattern}': {e}")
                    continue
            if value:
                try:
                    if field == "amount":
                        value = float(value.replace(",", ""))
                    elif field == "transaction_date":
                        value = self._normalize_date(value)
                    extracted_data[field] = value
                except ValueError as e:
                    logger.warning(f"Value conversion failed for field '{field}' with value '{value}': {e}")
        return extracted_data

    def validate_classification(self, content: str, request_type: str, sub_type: str, extracted_data: Dict) -> float:
        self._initialize_classifiers()
        expected_fields = REQUEST_TYPES.get(request_type, {}).get("fields", [])
        field_match_score = sum(1 for f in expected_fields if f in extracted_data) / len(expected_fields) if expected_fields else 0
        sentiment = self.sentiment_classifier(content)[0]
        sentiment_boost = 0.1 if sentiment["label"] == "NEGATIVE" and "Fee" in request_type else 0.0
        return min(1.0, field_match_score + sentiment_boost)

    def _split_into_enquiries(self, content: str) -> List[str]:
        segments = re.split(r'\n{2,}|\n\s*[-*]\s*|Subject:', content, flags=re.IGNORECASE)  # Reduced reliance on \n
        enquiries = [seg.strip() for seg in segments if seg.strip() and len(seg.strip()) > 20]
        return enquiries or [content.strip()]

    def _classify_single_enquiry(self, content: str) -> Dict[str, any]:
        if not content.strip():
            return {"request_type": "NA", "sub_request_type": "NA", "reason": "No meaningful content found.", "confidence": 0.0, "extracted_data": {}, "enquiry_text": ""}
        
        self._initialize_classifiers()
        try:
            main_types = list(REQUEST_TYPES.keys())
            main_descriptions = ", ".join(f"- {k}: {v['description']}" for k, v in REQUEST_TYPES.items())
            main_prompt = self._build_main_prompt(content, main_descriptions)
            main_result = self.zero_shot_classifier(main_prompt, main_types)
            top_main = main_result["labels"][0]
            main_confidence = main_result["scores"][0]

            sub_types = REQUEST_TYPES[top_main].get("subcategories", {})
            if sub_types:
                sub_labels = list(sub_types.keys())
                sub_descriptions = ", ".join(f"- {k}: {v}" for k, v in sub_types.items())
                sub_prompt = self._build_sub_prompt(content, top_main, sub_descriptions)
                sub_result = self.zero_shot_classifier(sub_prompt, sub_labels)
                top_sub = sub_result["labels"][0]
                sub_confidence = sub_result["scores"][0]
                extracted_data = self.extract_fields(content, top_main)
                base_confidence = min(main_confidence, sub_confidence)
                validation_boost = self.validate_classification(content, top_main, top_sub, extracted_data)
                final_confidence = round(base_confidence * 0.7 + validation_boost * 0.3, 4)
                result = {
                    "request_type": top_main,
                    "sub_request_type": top_sub,
                    "reason": sub_types[top_sub],
                    "confidence": final_confidence,
                    "extracted_data": extracted_data,
                    "enquiry_text": content[:200]
                }
                if final_confidence < self.confidence_threshold:
                    result["warning"] = "Low confidence; manual review recommended."
                return result
            
            extracted_data = self.extract_fields(content, top_main)
            validation_boost = self.validate_classification(content, top_main, "NA", extracted_data)
            final_confidence = round(main_confidence * 0.7 + validation_boost * 0.3, 4)
            result = {
                "request_type": top_main,
                "sub_request_type": "NA",
                "reason": REQUEST_TYPES[top_main]["description"],
                "confidence": final_confidence,
                "extracted_data": extracted_data,
                "enquiry_text": content[:200]
            }
            if final_confidence < self.confidence_threshold:
                result["warning"] = "Low confidence; manual review recommended."
            return result
        except Exception as e:
            logger.error(f"Classification error for content '{content[:200]}': {e}")
            return {"request_type": "NA", "sub_request_type": "NA", "reason": f"Classification error: {str(e)}", "confidence": 0.0, "extracted_data": {}, "enquiry_text": content[:200]}

    def classify_email(self, email_content: str, attachment_content: str = "") -> List[Dict[str, any]]:
        content = email_content.strip()
        if not content and attachment_content.strip():
            content = attachment_content.strip()
        elif content and attachment_content.strip():
            content += f" --- Attachment Content --- {attachment_content}"  # Reduced \n usage
        
        if not content:
            return [{"request_type": "NA", "sub_request_type": "NA", "reason": "No meaningful content found.", "confidence": 0.0, "extracted_data": {}, "enquiry_text": ""}]
        
        enquiries = self._split_into_enquiries(content)
        classifications = [self._classify_single_enquiry(enquiry) for enquiry in enquiries if self._classify_single_enquiry(enquiry)["request_type"] != "NA" or len(enquiries) == 1]
        return classifications or [{"request_type": "NA", "sub_request_type": "NA", "reason": "No meaningful enquiries identified.", "confidence": 0.0, "extracted_data": {}, "enquiry_text": content[:200]}]

    def _build_main_prompt(self, content: str, descriptions: str) -> str:
        return f"You are an expert financial email classifier. Classify the email into one of these categories based on its intent and content: {descriptions} Examples: - 'Reallocate $500 in fees' → Adjustment - 'Disburse $10,000 in foreign currency' → Money Movement-Outbound Email Content: --- {content} --- Return the most likely category."

    def _build_sub_prompt(self, content: str, main_type: str, descriptions: str) -> str:
        return f"Given the main category '{main_type}', classify the specific request type. Subcategories: {descriptions} Examples: - 'Adjusting $500 fees' → Reallocation Fees (if Adjustment) - 'Sending $2000 abroad' → Foreign Currency (if Outbound) Email Content: --- {content} --- Return the most likely subcategory."

def process_directory(directory: str, processor: DocumentProcessor) -> List[Dict]:
    directory = Path(directory).expanduser()
    results = []
    for file_path in directory.iterdir():
        if file_path.suffix.lower() == ".pdf":
            text = processor.extract_text_from_pdf(file_path)
            classifications = processor.classify_email(text)
            results.append({"file": file_path.name, "classifications": classifications})
        elif file_path.suffix.lower() == ".eml":
            email_text, attachment_text = processor.extract_text_from_eml(file_path)
            classifications = processor.classify_email(email_text, attachment_text)
            results.append({"file": file_path.name, "classifications": classifications})
        else:
            logger.warning(f"Skipping unsupported file: {file_path.name}")
    return results

def main():
    processor = DocumentProcessor(confidence_threshold=0.75)
    email_directory = "/Users/paramita.santra/impks/hackhive-2025/emails" #"~/Desktop/hackhive-2025-main/emails"
    output_file = "classification_results.json"
    try:
        results = process_directory(email_directory, processor)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Classification completed! Results saved in {output_file}")
        logger.info("Classification Results: %s", json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()