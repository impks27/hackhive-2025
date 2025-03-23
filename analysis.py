import json
import ollama
import fitz  # PyMuPDF for reading PDFs
import os
import re
import mailparser
from typing import Dict, List, Optional, Tuple
import pdfplumber
from datetime import datetime


class FileReader:
    """Handles reading content from various file types."""

    def read_text_file(self, filename: str) -> str:
        """Reads content from a text file."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def extract_text_from_eml(self, eml_path: str) -> Tuple[str, str]:
        """Extracts text from an EML file, including attachments."""
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


class DataExtractor:
    """Extracts text from files in a data folder."""

    def __init__(self, folder: str):
        self.folder = folder
        self.reader = FileReader()

    def extract_text(self) -> str:
        """Extracts text from PDFs and EMLs in the folder."""
        if not os.path.exists(self.folder):
            print(f"Warning: Folder '{self.folder}' not found.")
            return ""

        extracted_text = []
        for filename in os.listdir(self.folder):
            file_path = os.path.join(self.folder, filename)
            if filename.lower().endswith(".pdf"):
                try:
                    with fitz.open(file_path) as doc:
                        text = " ".join([page.get_text("text").replace("\n", " ") for page in doc]).strip()
                        extracted_text.append(f"[{filename}]: {text}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            elif filename.lower().endswith(".eml"):
                email_text, attachment_text = self.reader.extract_text_from_eml(file_path)
                extracted_text.append(f"[{filename}]: {email_text}")
                if attachment_text:
                    extracted_text.append(f"\n--- Attachment Content --- {attachment_text}\n")
            elif filename.lower().endswith((".doc", ".docx")):
                print(f"TODO: Implement .doc/.docx support for {filename}")
        return "\n".join(extracted_text) if extracted_text else ""


class PromptBuilder:
    """Builds the prompt from static files and extracted data."""

    def __init__(self, objective_file: str, categories_file: str, instructions_file: str, data_folder: str, output_file: str):
        self.reader = FileReader()
        self.extractor = DataExtractor(data_folder)
        self.objective_file = objective_file
        self.categories_file = categories_file
        self.instructions_file = instructions_file
        self.output_file = output_file

    def build_prompt(self) -> str:
        """Constructs and saves the full prompt."""
        objective = self.reader.read_text_file(self.objective_file)
        categories = self.reader.read_text_file(self.categories_file)
        instructions = self.reader.read_text_file(self.instructions_file)
        email_to_classify = self.extractor.extract_text()

        prompt = f"{objective}\n\n{categories}\n\nEmail to Classify:\n{email_to_classify}\n\n{instructions}"
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        return prompt


class Classifier:
    """Handles classification and sub-classification using an AI model."""

    def __init__(self, model: str = "deepseek-r1:14b"):
        self.model = model
        self.reader = FileReader()

    def extract_json_block(self, text: str) -> Optional[str]:
        """Extracts JSON block from text."""
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        return match.group(1) if match else None

    def classify(self, prompt: str) -> List[Dict]:
        """Sends prompt to the model and processes the response."""
        print("🤖 Gearing up the AI engine... Compiling the classification request!")
        response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
        response_content_text = response['message']['content']
        response_content = self.extract_json_block(response_content_text)

        if not response_content:
            print("❌ ERROR: No valid JSON found in response.")
            return []

        print("📊 Data processed! Here’s the classified breakdown:")
        print(response_content)
        return json.loads(response_content)

    def sub_classify(self, category: str, associated_text: str, confidence_score: float, extracted_fields: Dict) -> List[Dict]:
        """Performs sub-classification for specific categories."""
        final_output = []
        sub_categories_file_mapping = self.reader.read_text_file("resources/ruleset_files.json")
        mapping = json.loads(sub_categories_file_mapping) if sub_categories_file_mapping else {}

        sub_categories_file_name = mapping.get(category)
        if not sub_categories_file_name:
            final_output.append({
                "category": category,
                "confidence_score": confidence_score,
                "sub_category": {},
                "extracted_fields": extracted_fields
            })
            return final_output

        sub_categories = self.reader.read_text_file(sub_categories_file_name)
        sub_objective = self.reader.read_text_file("resources/sub_objective.txt")
        sub_instructions = self.reader.read_text_file("resources/sub_instructions.txt")

        prompt_sub = f"{sub_objective}\n\n{sub_categories}\n\nEmail to Classify:\n{associated_text}\n\n{sub_instructions}"
        response_sub = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt_sub}])
        response_sub_content_text = response_sub['message']['content']
        response_sub_content = self.extract_json_block(response_sub_content_text)

        print("📊 Sub-classification processed! Here’s the breakdown:")
        print(response_sub_content)

        try:
            sub_response = json.loads(response_sub_content)
            final_output.append({
                "category": category,
                "confidence_score": confidence_score,
                "sub_category": {
                    "name": sub_response.get("category", ""),
                    "confidence_score": sub_response.get("confidence_score", 0)
                },
                "extracted_fields": extracted_fields
            })
        except json.JSONDecodeError:
            print("❌ ERROR: Sub-response is not valid JSON.")
            print(response_sub_content)
            final_output.append({
                "category": category,
                "confidence_score": confidence_score,
                "sub_category": {},
                "extracted_fields": extracted_fields
            })

        return final_output


class AnalysisApp:
    """Main application class coordinating the analysis process."""

    def __init__(self):
        self.prompt_builder = PromptBuilder(
            objective_file="resources/objective.txt",
            categories_file="resources/categories.txt",
            instructions_file="resources/instructions.txt",
            data_folder="data",
            output_file="resources/request.txt"
        )
        self.classifier = Classifier(model="deepseek-r1:14b")

    def run(self):
        """Runs the full analysis process."""
        prompt = self.prompt_builder.build_prompt()
        print(f"📂 Final prompt saved to {self.prompt_builder.output_file}")
        print("🚀 Sending the prompt to the AI model... Stand by for classification!")

        classifications = self.classifier.classify(prompt)
        final_output = []

        for item in classifications:
            category = item["classification"]["category"]
            confidence_score = item["classification"]["confidence_score"]
            associated_text = item.get("associated_text", "No associated text found")
            extracted_fields = item.get("extracted_fields", {})
            
            print(f"📌 {category}: {associated_text}")
            sub_results = self.classifier.sub_classify(category, associated_text, confidence_score, extracted_fields)
            final_output.extend(sub_results)

        print("Here's the final output. Enjoy!")
        print(json.dumps(final_output, indent=4))


if __name__ == "__main__":
    app = AnalysisApp()
    app.run()