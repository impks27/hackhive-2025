import streamlit as st
import os
import shutil
import pandas as pd 
import numpy as np
from classifier import AnalysisLauncher

class LendingServiceApp:
    def __init__(self):
        self.file_paths = []
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def analyze_files(self):
        output_dict = {}
        for filename in os.listdir(self.temp_dir):    
            engine = AnalysisLauncher(self.file_paths)
            output = engine.process(filename)
            output_dict[filename] = output
        return output_dict
    
    def clean_inventory(self):
        if os.path.exists(self.temp_dir) and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Deleted folder: {self.temp_dir}")

    def flatten_output(self, output_dict):
        data = []
        for file_name, entries in output_dict.items():
            for entry in entries:
                row = {
                    "File Name": file_name,
                    "Category": entry["category"],
                    "Category Confidence": entry["confidence_score"],
                    "Sub-Category": entry["sub_category"].get("name", "N/A"),
                    "Sub-Category Confidence": entry["sub_category"].get("confidence_score", np.nan),
                    "Deal Name": entry["extracted_fields"].get("deal_name", "N/A"),
                    "Amount": entry["extracted_fields"].get("amount", np.nan),
                    "Transaction Date": entry["extracted_fields"].get("transaction_date", "N/A"),
                    "Account Number": entry["extracted_fields"].get("account_number", "N/A"),
                    "Currency": entry["extracted_fields"].get("currency", "N/A"),
                }
                data.append(row)
        return pd.DataFrame(data)

    def run(self):
        st.title("Commercial Bank Lending Service")
        st.write("Choose how to provide files for analysis:")
        
        option = st.radio("Select an option:", ("Upload Files", "Specify Folder Path"))
        
        try:
            if option == "Upload Files":
                uploaded_files = st.file_uploader("Upload files to analyze", accept_multiple_files=True, type=["txt", "pdf", "eml"], help="Upload up to 200MB of files.")
                
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(self.temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        self.file_paths.append(temp_path)
                    st.success(f"Uploaded {len(uploaded_files)} file(s)")
            
            elif option == "Specify Folder Path":
                folder_path = st.text_input("Enter the folder path containing files to analyze:")
                if folder_path and os.path.isdir(folder_path):
                    self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                    st.success(f"Found {len(self.file_paths)} file(s) in folder: {folder_path}")
                else:
                    st.error("Invalid folder path. Please enter a valid directory.")
            
            if self.file_paths and st.button("Analyze"):
                with st.spinner("Running analysis..."):
                    result = self.analyze_files()
                    st.write(result)
                    df = self.flatten_output(result)
                    st.dataframe(df)
                self.clean_inventory()
            
            if not self.file_paths:
                st.info("Please upload files or specify a valid folder path to enable the 'Analyze' button.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app = LendingServiceApp()
    app.run()
