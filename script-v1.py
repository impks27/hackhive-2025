import json
import ollama
import fitz  # PyMuPDF for reading PDFs
import os

# Define file paths for different sections
objective_file = "resources/objective.txt"
categories_file = "resources/categories.txt"
instructions_file = "resources/instructions.txt"
data_folder = "data"  # Folder containing PDFs
request_file = "resources/request.txt"  # Output file for the final prompt

# Function to read content from a text file
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
        return ""

# Function to extract text from all PDFs in the data folder (single line per PDF)
def extract_text_from_pdfs(folder):
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

    return "\n".join(extracted_text) if extracted_text else ""

# Read static sections
objective = read_file(objective_file)
categories = read_file(categories_file)
instructions = read_file(instructions_file)

# Extract email content from PDFs
email_to_classify = extract_text_from_pdfs(data_folder)

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
response = ollama.chat(model="0xroyce/plutus:latest", messages=[{'role': 'user', 'content': prompt}]) #deepseek-r1:14b

# Print the response content
print("📊 Data processed! Here’s the classified breakdown:")
response_output = response['message']['content']
print(response_output)

# Loop through response and print associated text for each category
for item in json.loads(response_output): 
    category = item["classification"]["category"]
    if (category == "Money Movement - Inbound"):
        associated_text = item["associated_text"]
        print(f"📌 {category}: {associated_text}")
        sub_categories = read_file("resources/sub-money-movement-inbound.txt")
        sub_objectice = read_file("resources/sub_objective.txt")
        sub_instructions = read_file("resources/sub_instructions.txt")
        prompt_sub = f"{sub_objectice}\n\n{sub_categories}\n\nEmail to Classify:\n{associated_text}\n\n{sub_instructions}"
        print(f"📂 Here's the sub prompt for:")
        print(prompt_sub)
        response_sub = ollama.chat(model="0xroyce/plutus:latest", messages=[{'role': 'user', 'content': prompt_sub}]) #deepseek-r1:14b
        print("📊 Data processed! Here’s the classified breakdown:")
        response_output = response_sub['message']['content']
        print(response_output)