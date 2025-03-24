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
        if (filename.lower().endswith(".doc") or filename.lower().endswith(".doc")):  # Process only doc files
                print("Get docs")
        if filename.lower().endswith(".eml"):  # Process only eml files
                email_text, attachment_text = self.extract_text_from_eml(filename)
                print(f"email_text:",email_text)
                print(f"attachment_text:",attachment_text)
                extracted_text.append(f"[{filename}]: {email_text}")
                extracted_text.append(f"\n--- Attachment Content --- {attachment_text}\n")
                
        return "\n".join(extracted_text) if extracted_text else ""        
    
    # Function to extract text from all PDFs in the data folder (single line per PDF)
    def extract_text_from_data_folder(self, folder):
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
    
#     @staticmethod
#     def read_file(filename: str) -> str:
#         try:
#             with open(filename, "r", encoding="utf-8") as f:
#                 return f.read().strip()
#         except FileNotFoundError:
#             print(f"Warning: {filename} not found.")
#             return ""

# class TextExtractor:
#     @staticmethod
#     def extract_json_block(text: str) -> Optional[str]:
#         match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
#         return match.group(1) if match else None
    
#     @staticmethod
#     def extract_text_from_pdf(pdf_path: str) -> str:
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 return " ".join(page.extract_text() or "" for page in pdf.pages)
#         except Exception as e:
#             print(f"Error extracting text from {pdf_path}: {e}")
#             return ""
    
#     @staticmethod
#     def extract_text_from_eml(eml_path: str) -> Tuple[str, str]:
#         try:
#             mail = mailparser.parse_from_file(eml_path)
#             email_body = f"Subject: {mail.subject} Body: {mail.body}"
#             attachment_text = ""
#             for attachment in mail.attachments:
#                 content_type = attachment.get("content_type", "").lower()
#                 payload = attachment.get("payload", "")
#                 if "text/plain" in content_type:
#                     attachment_text += f" Attachment Text: {payload}"
#                 elif "application/pdf" in content_type:
#                     import base64
#                     try:
#                         pdf_data = base64.b64decode(payload)
#                         with open("temp.pdf", "wb") as f:
#                             f.write(pdf_data)
#                         attachment_text += f" Attachment PDF: {TextExtractor.extract_text_from_pdf('temp.pdf')}"
#                         os.remove("temp.pdf")
#                     except Exception as e:
#                         print(f"Error processing PDF attachment in {eml_path}: {e}")
#             return email_body, attachment_text
#         except Exception as e:
#             print(f"Error extracting text from {eml_path}: {e}")
#             return "", ""

# class AIClassifier:
#     def __init__(self, model_name: str):
#         self.model_name = model_name
    
#     def classify(self, prompt: str) -> str:
#         response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
#         return response['message']['content']

# class ClassificationEngine:
#     def __init__(self, data_folder: str, model_name: str):
#         self.data_folder = data_folder
#         self.ai_classifier = AIClassifier(model_name)
#         self.final_output = []
    
#     def process_file(self, file_path: str):
#         filename = os.path.basename(file_path)
#         print(f"Processing file: {filename}")
#         text = ""
        
#         if filename.lower().endswith(".pdf"):
#             text = TextExtractor.extract_text_from_pdf(file_path)
#         elif filename.lower().endswith(".eml"):
#             email_text, attachment_text = TextExtractor.extract_text_from_eml(file_path)
#             text = f"{email_text}\n{attachment_text}"
        
#         if not text:
#             print(f"Skipping {filename}, no text extracted.")
#             return
        
#         print(f"Extracted text from {filename}: {text[:500]}...")
        
#         objective = FileManager.read_file("resources/objective.txt")
#         categories = FileManager.read_file("resources/categories.txt")
#         instructions = FileManager.read_file("resources/instructions.txt")
        
#         prompt = f"{objective}\n\n{categories}\n\nEmail to Classify:\n{text}\n\n{instructions}"
#         response_content = TextExtractor.extract_json_block(self.ai_classifier.classify(prompt))
        
#         print(f"AI response for {filename}: {response_content}")
        
#         if not response_content:
#             print(f"Error: Invalid JSON response for {filename}.")
#             return
        
#         try:
#             response_data = json.loads(response_content)
#         except json.JSONDecodeError:
#             print(f"❌ ERROR: AI response is not valid JSON for {filename}.")
#             return
        
#         for item in response_data:
#             category = item["classification"]["category"]
#             confidence_score = item["classification"]["confidence_score"]
#             extracted_fields = item.get("extracted_fields", [])
            
#             sub_objective = FileManager.read_file("resources/sub_objective.txt")
#             sub_categories_mapping = json.loads(FileManager.read_file("resources/ruleset_files.json"))
#             sub_categories = sub_categories_mapping.get(category, "")
#             sub_instructions = FileManager.read_file("resources/sub_instructions.txt")
            
#             sub_prompt = f"{sub_objective}\n\n{sub_categories}\n\nEmail to Classify:\n{text}\n\n{sub_instructions}"
#             sub_response_content = TextExtractor.extract_json_block(self.ai_classifier.classify(sub_prompt))
            
#             print(f"Sub-classification response for {filename}: {sub_response_content}")
            
#             if sub_response_content:
#                 try:
#                     sub_classification = json.loads(sub_response_content)
#                     self.final_output.append({
#                         "file": filename,
#                         "category": category,
#                         "confidence_score": confidence_score,
#                         "sub_category": {
#                             "name": sub_classification["category"],
#                             "confidence_score": sub_classification["confidence_score"]
#                         },
#                         "extracted_fields": extracted_fields
#                     })
#                 except json.JSONDecodeError:
#                     print(f"❌ ERROR: Sub-response is not valid JSON for {filename}.")
#             else:
#                 self.final_output.append({
#                     "file": filename,
#                     "category": category,
#                     "confidence_score": confidence_score,
#                     "sub_category": {},
#                     "extracted_fields": extracted_fields
#                 })
    
    def process(self, filename):
        print(f"Processing file {filename}...")
#         if not os.path.exists(self.data_folder):
#             print(f"Warning: Folder '{self.data_folder}' not found.")
#             return json.dumps([])
        
#         for filename in os.listdir(self.data_folder):
#             file_path = os.path.join(self.data_folder, filename)
#             self.process_file(file_path)
        
#         return json.dumps(self.final_output, indent=4)
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
        #all_files_output = [filemale, final_output]
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

            # ✅ Only process "Money Movement - Inbound"
            #if category == "Money Movement - Inbound":
                #sub_categories = self.read_file("resources/sub-category-money-movement-inbound.txt")
                sub_objective = self.read_file("resources/sub_objective.txt")
                sub_instructions = self.read_file("resources/sub_instructions.txt")

                prompt_sub = f"{sub_objective}\n\n{sub_categories}\n\nEmail to Classify:\n{associated_text}\n\n{sub_instructions}"
                
                #print("📂 Here's the sub prompt for further classification:")
                #print(prompt_sub)

                #🔥 Send sub-classification request
                response_sub = ollama.chat(model="deepseek-r1:14b", messages=[{'role': 'user', 'content': prompt_sub}]) 

                print("📊 Sub-classification processed! Here’s the breakdown:")
                response_sub_content_text = response_sub['message']['content']
                response_sub_content = self.extract_json_block(response_sub_content_text)

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
        return final_output 

if __name__ == "__main__":
    engine = AnalysisLauncher("temp")
    output = engine.process()
    print(output)
