import json
import ollama
import fitz  # PyMuPDF for reading PDFs
import os
import re
import mailparser
from typing import Dict, List, Optional, Tuple
import pdfplumber
import mailparser
from transformers import pipeline
from datetime import datetime

# Define file paths for different sections
objective_file = "resources/objective.txt"
categories_file = "resources/categories.txt"
instructions_file = "resources/instructions.txt"
data_folder = "data"  # Folder containing PDFs
request_file = "resources/request.txt"  # Output file for the final prompt



# Function to read content from a text file
def extract_json_block(text):
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    return match.group(1) if match else None

def extract_text_from_pdf(self, pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = " ".join(page.extract_text() or "" for page in pdf.pages)  # Replaced \n with space
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
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
                        attachment_text += f" Attachment PDF: {extract_text_from_pdf('temp.pdf')}"
                        os.remove("temp.pdf")
                    except Exception as e:
                        print(f"Error processing PDF attachment in {eml_path}: {e}")
            return email_body, attachment_text
        except Exception as e:
            print(f"Error extracting text from {eml_path}: {e}")
            return "", ""

# Function to extract text from all PDFs in the data folder (single line per PDF)
def extract_text_from_data_folder(folder):
    extracted_text = []
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' not found.")
        return ""

    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):  # Process only PDF files
            pdf_path = os.path.join(folder, filename)
            try:
                with fitz.open(pdf_path) as doc:
                    text = " ".join([page.get_text("text").replace("\n", " ") for page in doc]).strip()
                    extracted_text.append(f"[{filename}]: {text}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        if (filename.lower().endswith(".doc") or filename.lower().endswith(".doc")):  # Process only doc files
            print("Get docs")
        if filename.lower().endswith(".eml"):  # Process only eml files
            email_text, attachment_text = extract_text_from_eml(filename)
            print(f"email_text:",email_text)
            print(f"attachment_text:",attachment_text)
            extracted_text.append(f"[{filename}]: {email_text}")
            extracted_text.append(f"\n--- Attachment Content --- {attachment_text}\n")
            
            
    return "\n".join(extracted_text) if extracted_text else ""

# Read static sections
objective = read_file(objective_file)
categories = read_file(categories_file)
instructions = read_file(instructions_file)
final_output = []
# Extract email content from PDFs
email_to_classify = extract_text_from_data_folder(data_folder)

# Combine all sections into the final prompt
prompt = f"{objective}\n\n{categories}\n\nEmail to Classify:\n{email_to_classify}\n\n{instructions}"

# Save the prompt to a file
os.makedirs(os.path.dirname(request_file), exist_ok=True)  # Ensure the folder exists
with open(request_file, "w", encoding="utf-8") as f:
    f.write(prompt)

print("🤖 Gearing up the AI engine... Compiling the classification request!")
print(f"📂 Final prompt saved to {request_file}")
print("🚀 Sending the prompt to the AI model... Stand by for classification!")

# Send the prompt to the model
response = ollama.chat(model="deepseek-r1:14b", messages=[{'role': 'user', 'content': prompt}]) #deepseek-r1:14b, 0xroyce/plutus:latest

# Print the response content
print("📊 Data processed! Here’s the classified breakdown:")
response_content_text = response['message']['content']
response_content = extract_json_block(response_content_text)
classification_response = response_content
print(response_content)
sub_classification_response = []
#🔄 Loop through response and process categories
for item in json.loads(response_content): 
    category = item["classification"]["category"]
    confidence_score = item["classification"]["confidence_score"]
    associated_text = item.get("associated_text", "No associated text found.")
    extracted_fields = item.get("extracted_fields",[])
    
    print(f"📌 {category}: {associated_text}")
    
    # Load ruleset
    sub_categories_file_mapping = read_file("resources/ruleset_files.json")
    sub_categories_file_name = json.loads(sub_categories_file_mapping)[category]
    if sub_categories_file_name:
        sub_categories = read_file(sub_categories_file_name)

    # ✅ Only process "Money Movement - Inbound"
    #if category == "Money Movement - Inbound":
        #sub_categories = read_file("resources/sub-category-money-movement-inbound.txt")
        sub_objective = read_file("resources/sub_objective.txt")
        sub_instructions = read_file("resources/sub_instructions.txt")

        prompt_sub = f"{sub_objective}\n\n{sub_categories}\n\nEmail to Classify:\n{associated_text}\n\n{sub_instructions}"
        
        #print("📂 Here's the sub prompt for further classification:")
        #print(prompt_sub)

        #🔥 Send sub-classification request
        response_sub = ollama.chat(model="deepseek-r1:14b", messages=[{'role': 'user', 'content': prompt_sub}]) 

        print("📊 Sub-classification processed! Here’s the breakdown:")
        response_sub_content_text = response_sub['message']['content']
        response_sub_content = extract_json_block(response_sub_content_text)

        # ✅ Ensure sub-response is valid JSON
        try:
            #response_sub_output = json.loads(response_sub_content)  # Convert string to JSON
            #print(json.dumps(response_sub_content, indent=4))  # Pretty print the JSON output
            print("Here the sub category response with confidence score:")
            print(response_sub_content)
            #sub_classification_response.append(response_sub_content)
            sub_classification_response_json = json.loads(response_sub_content)
            sub_category_name = sub_classification_response_json["category"]
            sub_confidence_score = sub_classification_response_json["confidence_score"]
            final_output.append({
                    "category": category,
                    "confidence_score": confidence_score,
                    "sub_category": {
                        "name": sub_category_name,
                        "confidence_score": sub_confidence_score
                    },
                    "extracted_fields": extracted_fields
                })           
        except json.JSONDecodeError:
            print("❌ ERROR: Sub-response is not valid JSON.")
            print(response_sub_content)
    else:
        final_output.append({
                    "category": category,
                    "confidence_score": confidence_score,
                    "sub_category": {},
                    "extracted_fields": extracted_fields
                }) 
# Switching to deep seek after this

# Generate the final structured output
# Convert to JSON format and print
print("Here's the final output. Enjoy!")
print(json.dumps(final_output, indent=4))


