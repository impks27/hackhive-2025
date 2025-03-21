import traceback
import json
from langchain_community.document_loaders import PyMuPDFLoader
from llama_cpp import Llama  

# Path to the downloaded GGML model
MODEL_PATH = r"/Users/paramita.santra/impks/hackhive-2025/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_0.bin"

try:
    print("🚀 Loading the LLaMA model...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=512, verbose=True)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading the model!")
    print(traceback.format_exc())
    exit(1)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    except Exception as e:
        print("❌ Error extracting text from PDF!")
        print(traceback.format_exc())
        return ""

# Function to classify the email
def classify_email(pdf_path):
    email_text = extract_text_from_pdf(pdf_path)

    if not email_text.strip():
        print("⚠️ No text extracted from PDF. Cannot classify.")
        return {"request_type": "NA", "reason": "No text found in PDF"}

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
    - Contract Renewal Request

    Also, provide reasoning for your classification.

    Respond in JSON format:
    {{
        "request_type": "<request type>",
        "reason": "<explanation>"
    }}
    """

    try:
        test_prompt = "Tell me a joke individually and in english"
        test_response = llm(test_prompt, max_tokens=50)
        print("🛠️ Model Test Response Joke:", test_response)

        test_response = llm("What is 2 + 2?", max_tokens=256, echo=True)
        print("🔍 Model Test Response:", test_response)

        response = llm(prompt, max_tokens=256,  echo=True)
        print("response----------------")
        print(response)

        if "choices" not in response or not response["choices"]:
            print("⚠️ Model response is empty or malformed.")
            return {"request_type": "NA", "reason": "No valid response from model"}

        result = response["choices"][0].get("text", "").strip()

        if not result:
            print("⚠️ Model did not return a classification.")
            return {"request_type": "NA", "reason": "No classification found in response"}

        print("result------------------")
        print(result)

        # Try parsing as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON format from model response.")
            return {"request_type": "NA", "reason": "Failed to parse model output"}

    except Exception as e:
        print("❌ Error during classification!")
        print(traceback.format_exc())
        return {"request_type": "NA", "reason": "Unexpected error occurred"}

# Run classification
pdf_path = r"/Users/paramita.santra/impks/hackhive-2025/sample_email.pdf"
classification = classify_email(pdf_path)

print("🔹 Final Classification Result:")
print(json.dumps(classification, indent=2))

