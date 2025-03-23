import streamlit as st
import os
import subprocess
from typing import List
import json
import ollama
import fitz  # PyMuPDF for reading PDFs
import pdfplumber
import mailparser
import re
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np


# From your previous script: FileReader, DataExtractor, PromptBuilder, Classifier
class FileReader:
    """Handles reading content from various file types."""

    def read_text_file(self, filename: str) -> str:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def extract_text_from_eml(self, eml_path: str) -> tuple[str, str]:
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

    def extract_json_block(self, text: str) -> str | None:
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        return match.group(1) if match else None

    def classify(self, prompt: str) -> List[dict]:
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

    def sub_classify(self, category: str, associated_text: str, confidence_score: float, extracted_fields: dict) -> List[dict]:
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

    def run(self) -> str:
        """Runs the full analysis process and returns the result as a JSON string."""
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

        result = json.dumps(final_output, indent=4)
        print("Here's the final output. Enjoy!")
        print(result)
        return result


# Streamlit OOP Classes
class FileHandler:
    """Handles file uploads and folder path processing."""

    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def save_uploaded_files(self, uploaded_files) -> List[str]:
        """Saves uploaded files to a temporary directory and returns their paths."""
        file_paths = []
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_path)
        return file_paths

    def get_files_from_folder(self, folder_path: str) -> List[str]:
        """Returns a list of file paths from a specified folder."""
        if os.path.isdir(folder_path):
            return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return []


class StreamlitApp:
    """Main Streamlit application class."""

    def __init__(self):
        self.file_handler = FileHandler()
        self.analysis_app = AnalysisApp()

    def analyze_files(self, file_paths: List[str]) -> str:
        """Analyzes files by invoking AnalysisApp and returns the result."""
        st.write("Analyzing files using Deepseek...")
        
        # Move files to the data folder expected by AnalysisApp
        data_folder = "data"
        os.makedirs(data_folder, exist_ok=True)
        for file_path in file_paths:
            dest_path = os.path.join(data_folder, os.path.basename(file_path))
            os.rename(file_path, dest_path)  # Move file to data folder
        
        # Run the analysis
        try:
            result = self.analysis_app.run()
            return result
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def flatten_json_to_df(self, json_data: str) -> pd.DataFrame:
        """Flattens the nested JSON into a pandas DataFrame for table display."""
        data = json.loads(json_data)
        flattened_data = []
        
        for item in data:
            row = {
                "Category": item["category"],
                "Confidence Score": item["confidence_score"],
                "Sub-Category": item["sub_category"].get("name", "N/A"),
                "Sub-Category Confidence": item["sub_category"].get("confidence_score", np.nan),
                "Deal Name": item["extracted_fields"]["deal_name"],
                "Amount": item["extracted_fields"]["amount"],
                "Transaction Date": item["extracted_fields"]["transaction_date"],
                "Account Number": item["extracted_fields"]["account_number"],
                "Currency": item["extracted_fields"]["currency"]
            }
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)

    def run(self):
        """Runs the Streamlit UI."""
        st.title("Commercial Bank Lending Service")
        st.write("Choose how to provide files for analysis:")

        option = st.radio("Select an option:", ("Upload Files", "Specify Folder Path"))
        file_paths = []

        if option == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload files to analyze",
                accept_multiple_files=True,
                type=["txt", "pdf", "eml"],
                help="Upload up to 200MB of files (e.g., .txt, .pdf, .eml)"
            )
            if uploaded_files:
                file_paths = self.file_handler.save_uploaded_files(uploaded_files)
                st.success(f"Uploaded {len(uploaded_files)} file(s): {', '.join([f.name for f in uploaded_files])}")

        elif option == "Specify Folder Path":
            folder_path = st.text_input("Enter the folder path containing files to analyze:", "")
            if folder_path:
                file_paths = self.file_handler.get_files_from_folder(folder_path)
                if file_paths:
                    st.success(f"Found {len(file_paths)} file(s) in folder: {folder_path}")
                else:
                    st.error("Invalid folder path or no files found. Please enter a valid directory.")

        if file_paths and st.button("Analyze"):
            with st.spinner("Running analysis..."):
                result = self.analyze_files(file_paths)
                st.write("Analysis Result:")
                # Convert JSON string to DataFrame and display as table
                if result and not result.startswith("Error"):
                    df = self.flatten_json_to_df(result)
                    st.dataframe(df)  # Display as interactive table
                else:
                    st.error(result)

        if not file_paths:
            st.info("Please upload files or specify a valid folder path to enable the 'Analyze' button.")


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()