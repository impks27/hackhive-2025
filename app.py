import streamlit as st
import os
import time
import subprocess
from refactor_script_v1 import AnalysisLauncher
import pandas as pd 
import numpy as np

# Placeholder for your analysis script
def analyze_files(file_paths):
    # """
    # Replace this with your actual Python script logic to analyze the files.
    # For now, it simulates analysis and returns a dummy result.
    # """
    # st.write("Analyzing files using Deepseek...")
    # time.sleep(2)  # Simulate processing time
    # result = f"Analysis complete! Processed files: {', '.join(file_paths)}"
    # st.write("Analyzing files using Deepseek...")
    
    # # Construct the command to run script-v1.py with file paths as arguments
    # command = ["python3", "script-v1.py"] #+ file_paths
    
    # try:
    #     # Run the script and capture output
    #     result = subprocess.run(
    #         command,
    #         capture_output=True,
    #         text=True,
    #         check=True
    #     )
    #     output = result.stdout
    #     if result.stderr:
    #         st.error(f"Script errors: {result.stderr}")    
    #     return f"Analysis complete! Output from script-v1.py:\n{output}"
    # except subprocess.CalledProcessError as e:
    #     return f"Error running script-v1.py: {e.stderr}"
    # except Exception as e:
    #     return f"Unexpected error: {str(e)}"
    output_dict = {}
    for filename in os.listdir("temp"):    
        engine = AnalysisLauncher(file_paths)
        output = engine.process(filename)
        output_dict.update({filename: output})
        print(output_dict)
    
    return output_dict

# Streamlit app
def main():
    st.title("Commercial Bank Lending Service")
    st.write("Choose how to provide files for analysis:")

    # Option selection
    option = st.radio("Select an option:", ("Upload Files", "Specify Folder Path"))

    file_paths = []

    if option == "Upload Files":
        # File upload option
        uploaded_files = st.file_uploader(
            "Upload files to analyze",
            accept_multiple_files=True,
            type=["txt", "pdf"],  # Add more file types as needed
            help="Upload up to 200MB of files (e.g., .txt, .pdf)"
        )
        
        if uploaded_files:
            # Save uploaded files temporarily and collect paths
            file_paths = []
            for uploaded_file in uploaded_files:
                # Write to temporary file
                temp_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(temp_path)
            st.success(f"Uploaded {len(uploaded_files)} file(s): {', '.join([f.name for f in uploaded_files])}")

    elif option == "Specify Folder Path":
        # Folder path input option
        folder_path = st.text_input("Enter the folder path containing files to analyze:", "")
        if folder_path:
            if os.path.isdir(folder_path):
                file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                st.success(f"Found {len(file_paths)} file(s) in folder: {folder_path}")
            else:
                st.error("Invalid folder path. Please enter a valid directory.")

    # Analyze button
    if file_paths and st.button("Analyze"):
        with st.spinner("Running analysis..."):
            result = analyze_files("temp")
            st.write(result)
            df = flatten_output(result)
            st.dataframe(df)
    
    if not file_paths:
        st.info("Please upload files or specify a valid folder path to enable the 'Analyze' button.")

def flatten_output(output_dict):
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

if __name__ == "__main__":
    # Ensure temp directory exists for uploaded files
    os.makedirs("temp", exist_ok=True)
    main()