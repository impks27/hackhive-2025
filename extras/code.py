import openai
import pytesseract
import pdfplumber
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load NLP & Embeddings Model
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# OCR Processing
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_image(image_bytes):
    return pytesseract.image_to_string(image_bytes)

# LLM-Based Intent Classification
def classify_email(content):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in categorizing emails."},
                  {"role": "user", "content": f"Classify this email: {content}"}]
    )
    return response["choices"][0]["message"]["content"]

# Named Entity Recognition for Data Extraction
def extract_entities(text):
    doc = nlp(text)
    extracted = {ent.label_: ent.text for ent in doc.ents}
    return extracted

# Multi-Request Handling
def detect_multiple_requests(content):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Extract multiple request types from this email."},
                  {"role": "user", "content": content}]
    )
    return response["choices"][0]["message"]["content"]

# Duplicate Email Detection
def is_duplicate(email1, email2):
    embedding1 = embed_model.encode(email1, convert_to_tensor=True)
    embedding2 = embed_model.encode(email2, convert_to_tensor=True)
    similarity = cosine_similarity([embedding1.cpu().numpy()], [embedding2.cpu().numpy()])[0][0]
    return similarity > 0.85  # Threshold for duplicate detection

# Process Emails from PDF
def process_emails_from_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    category = classify_email(pdf_text)
    entities = extract_entities(pdf_text)
    multi_requests = detect_multiple_requests(pdf_text)
    
    print(f"Category: {category}")
    print(f"Extracted Entities: {entities}")
    print(f"Detected Requests: {multi_requests}")

# Run the pipeline with sample PDF
process_emails_from_pdf("sample_emails.pdf")

