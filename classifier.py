import json
import ollama
import fitz  # PyMuPDF for reading PDFs
import os
import re
import mailparser
import pdfplumber
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class AnalysisLauncher:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name 
    # Function to read content from a text file
    def extract_json_block(self, text):
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

    def read_file(self,filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")
            return ""

    def extract_text_from_eml(self, eml_path: str) -> Tuple[str, str]:
            try:
                mail = mailparser.parse_from_file(eml_path)
                email_body = f"{mail.body}"
                attachment_text = ""
                for attachment in mail.attachments:
                    content_type = attachment.get("mail_content_type", "application/pdf").lower()
                    payload = attachment.get("payload", "")
                    attachment_filename = attachment.get("filename", "temp.pdf")  # Get actual filename
                    print(f"attachment_filename:", attachment_filename)
                    print(f"attachment: {attachment}")
                    if "text/plain" in content_type:
                        attachment_text += f" Attachment Text: {payload}"
                    elif "application/pdf" in content_type:
                        import base64
                        try:
                            pdf_data = base64.b64decode(payload)
                            with open(attachment_filename, "wb") as f:
                                f.write(pdf_data)
                            attachment_text += f" Attachment PDF: {self.extract_text_from_pdf(attachment_filename)}"
                            os.remove(attachment_filename)
                        except Exception as e:
                            print(f"Error processing PDF attachment in {eml_path}: {e}")
                return email_body, attachment_text
            except Exception as e:
                print(f"Error extracting text from {eml_path}: {e}")
                return "", ""

    def extract_text_from_file(self, filename):
        extracted_text = []
        if filename.lower().endswith(".pdf"):  # Process only PDF files
                pdf_path = os.path.join("temp", filename)
                try:
                    with fitz.open(pdf_path) as doc:
                        text = " ".join([page.get_text("text").replace("\n", " ") for page in doc]).strip()
                        extracted_text.append(f"[{filename}]: {text}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        if (filename.lower().endswith(".doc") or filename.lower().endswith(".docx")):  # Process only doc files
                print("Get docs")
        if filename.lower().endswith(".eml"):  # Process only eml files
                eml_path = os.path.join("temp", filename)
                print(f"eml_path:",eml_path)
                email_text, attachment_text = self.extract_text_from_eml(eml_path)
                print(f"email_text:",email_text)
                print(f"attachment_text:",attachment_text)
                text = (email_text.replace("\n", " ")).replace("*","").strip()
                extracted_text.append(f"[{filename}]: {text}")
                if attachment_text:
                    attachment_text = (attachment_text.replace("\n", " ")).replace("*","").strip()
                    extracted_text.append(f"--- Attachment Content --- {attachment_text}")
                
        return "\n".join(extracted_text) if extracted_text else ""        
    
    def process(self, filename):
        print(f"Processing file {filename}...")
        # Define file paths for different sections
        objective_file = "resources/objective.txt"
        categories_file = "resources/categories.txt"
        instructions_file = "resources/instructions.txt"
        data_folder = self.folder_name  # Folder containing PDFs
        request_file = "resources/request.txt"  # Output file for the final prompt
        # Read static sections
        objective = self.read_file(objective_file)
        categories = self.read_file(categories_file)
        instructions = self.read_file(instructions_file)
        #all_files_output = []
        final_output = []
        # Extract email content from PDFs
        email_to_classify = self.extract_text_from_file(filename)

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
        response_content = self.extract_json_block(response_content_text)
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
            sub_categories_file_mapping = self.read_file("resources/ruleset_files.json")
            sub_categories_file_name = json.loads(sub_categories_file_mapping)[category]
            if sub_categories_file_name:
                sub_categories = self.read_file(sub_categories_file_name)
                sub_objective = self.read_file("resources/sub_objective.txt")
                sub_instructions = self.read_file("resources/sub_instructions.txt")

                prompt_sub = f"{sub_objective}\n\n{sub_categories}\n\nEmail to Classify:\n{associated_text}\n\n{sub_instructions}"

                #🔥 Send sub-classification request
                response_sub = ollama.chat(model="deepseek-r1:14b", messages=[{'role': 'user', 'content': prompt_sub}]) 

                print("📊 Sub-classification processed! Here’s the breakdown:")
                response_sub_content_text = response_sub['message']['content']
                response_sub_content = self.extract_json_block(response_sub_content_text)

                # ✅ Ensure sub-response is valid JSON
                try:
                    print("Here the sub category response with confidence score:")
                    print(response_sub_content)
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
        return final_output 

if __name__ == "__main__":
    engine = AnalysisLauncher("temp")
    output = engine.process()
    print(output)
