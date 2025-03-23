from langchain_community.document_loaders import PyMuPDFLoader
from llama_cpp import Llama  
import json

# Path to the downloaded GGML model
MODEL_PATH = r"/Users/paramita.santra/impks/hackhive-2025/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_0.bin"

# Load LLaMA model
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=512, verbose=True)  

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

# Function to classify the email
def classify_email(pdf_path):
    email_text = extract_text_from_pdf(pdf_path)
    print("-------------")
    print(email_text)
    prompt = f"""
    You are an AI that classifies emails into predefined request types.
    Given the following email:
    ---
    {email_text}
    ---
    
    Classify the email into one of the following request types:
    - Billing Issue
    - Technical Support
    - Account Management
    - General Inquiry

    Also, provide reasoning for your classification.

    Respond in JSON format:
    {{
        "request_type": "<request type>",
        "reason": "<explanation>"
    }}
    """

    response = llm(prompt, max_tokens=256)
    print("response----------------")
    print(response)
    result = response["choices"][0]["text"]

    try:
        print("result------------------")
        print(result)
        return json.loads(result)  # Convert response to JSON
    except json.JSONDecodeError:
        print(traceback.format_exc())
        return {"error": "Invalid response format", "raw_response": result}

# Run classification
pdf_path = r"/Users/paramita.santra/impks/hackhive-2025/sample_email.pdf"
classification = classify_email(pdf_path)
print(json.dumps(classification, indent=2))

