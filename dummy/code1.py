from langchain.document_loaders import PyMuPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

# Define classification prompt
classification_prompt = PromptTemplate(
    input_variables=["email_text"],
    template="""
    You are an AI that classifies emails into predefined categories.
    Given the following email text:
    ---
    {email_text}
    ---
    
    Classify it into one of the following request types: 
    - Billing Issue
    - Technical Support
    - Account Management
    - General Inquiry
    
    Also, provide a reason for the classification.
    
    Response format:
    {{
        "request_type": "<request type>",
        "reason": "<explanation>"
    }}
    """
)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Define LangChain LLMChain
classification_chain = LLMChain(
    llm=llm,
    prompt=classification_prompt
)

# Process email PDF
def classify_email(pdf_path):
    email_text = extract_text_from_pdf(pdf_path)
    response = classification_chain.run(email_text)
    return response

# Example usage
if __name__ == "__main__":
    pdf_path = "/Users/paramita.santra/impks/hackhive-2025/sample_email.pdf"  # Replace with actual file
    classification_result = classify_email(pdf_path)
    print(classification_result)

