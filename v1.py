import ollama
import json

# Define the prompt
prompt = """
Objective: Classify the provided email into the most appropriate categories based on the descriptions and sample emails provided for each category. The text can contain requests for multiple transactions each belonging to a distinct category. If the email contains text that belongs to multiple categories, create a JSON array where each object represents a distinct category and its associated fields and the text from the email that supports the classification. Include a confidence score for each classification.

Categories:
1. Category Name: Adjustment
    * Description: This is when we change something about a money deal that's already set up. It could be tweaking how much someone owes, updating fees, or fixing details in an agreement. It's like making small updates to keep things right.
    * Sample Email: Please adjust the fee structure for Deal XYZ to $5000 effective 03/15/2025.
2. Category Name: AU Transfer
    * Description: This is about moving money between different parts of a financial setup, called Allocation Units (AU). It's like shifting a chunk of cash from one bucket to another in the same system, usually the main amount someone borrowed.
    * Sample Email: Transfer $10,000 from Deal ABC to Deal DEF on 03/20/2025.
3. Category Name: Closing Notice
    * Description: This is when we tell people a money deal is ending or changing in a big way. It's like sending a heads-up that we're wrapping things up or tweaking something major, so everyone knows what's happening.
    * Sample Email: Issue a closing notice for Deal PQR with a $1000 adjustment on 03/19/2025.
4. Category Name: Commitment Change
    * Description: This is when we adjust how much money we've promised to give or hold for a loan or credit deal. It's like changing our pledge—maybe giving more, less, or just rolling it over without extra cash.
    * Sample Email: Adjust commitment for Deal BCD by $3,000 on 03/14/2025.
5. Category Name: Fee Payment
    * Description: This is when we handle payments for extra charges tied to money deals or loans. It's like collecting the costs for keeping things running or making special changes.
    * Sample Email: Submit a fee payment of $250 for Deal KLM on 03/15/2025, account 54321.
6. Category Name: Money Movement - Inbound
    * Description: This is when money comes into the bank from outside. It could be someone paying back a loan, sending interest, or covering fees—basically any cash flowing our way.
    * Sample Email: Receive $14,000 inbound for Deal MNO on 03/25/2025, account 77889.
7. Category Name: Money Movement - Outbound
    * Description: This is when money leaves the bank to go somewhere else. It includes things like sending out loans, paying someone, or moving funds to another account or place. Basically, any time the bank sends cash outward.
    * Sample Email: Transfer $16,500 for Deal YZA on 03/26/2025 in USD.

Email to Classify:
Loan Services Request Date: March 22, 2025 To: Loan Services Team Subject: Multiple Inbound Payment Requests for Project Delta Dear Team, Please process an inbound payment for Deal Name: Project Delta. Amount: $2,000. Transaction Date: 03/24/2025. Account Number: ACC98765. This payment covers the principal amount of the loan. Additionally, we have another inbound payment for the same deal. Amount: $500. Transaction Date: 03/25/2025. Account Number: ACC98765. This is to cover the interest accrued this month. Furthermore, we'd like to adjust the fee structure for Project Delta. Amount: $100. Transaction Date: 03/26/2025. This adjustment is due to a recent amendment in terms. Best regards, Alex Carter


Instructions:
1. Analyze the email content and assign it to the most appropriate categories based on the provided descriptions and sample emails.
2. If the email contains text that belongs to multiple categories, create a JSON array where each object represents a distinct category, its associated fields, and the text from the email that supports the classification.
3. Provide a confidence score for each classification between 0 and 1, where:
    * 0.9-1.0: Very confident
    * 0.7-0.89: Confident
    * 0.5-0.69: Somewhat confident
    * Below 0.5: Not confident
4. Extract below fields for each category if found in inputs else mark as NA:
    * deal_name
    * amount
    * transaction_date
    * account_number
    * currency
5. Include the complete text from the email that is considered for classification under the key "associated_text".
6. Provide a brief explanation for each classification.
7. Provide the output in the following JSON format:
[
    {
        "classification": {
            "category": "Category Name",
            "confidence_score": 0.95
        },
        "extracted_fields": {
            "deal_name": "NA",
            "amount": "$5,000.00",
            "transaction_date": "March 20, 2025",
            "account_number": "123456",
            "currency": "NA"
        },
        "associated_text": "We have successfully received your loan payment for Loan Account #123456 on March 20, 2025. Details of the transaction: - Amount Received: $5,000.00 - Payment Method: Wire Transfer - Reference Number: TXN56789",
        "explanation": "The email confirms the receipt of a loan payment, which aligns with the 'Money Movement - Inbound' category as it involves money coming into the bank from an external source. The confidence score is high (0.95) because the email clearly describes an inbound transaction and matches the description and sample emails provided for this category."
    },
    {
        "classification": {
            "category": "Adjustment",
            "confidence_score": 0.85
        },
        "extracted_fields": {
            "deal_name": "NA",
            "amount": "$200",
            "transaction_date": "NA",
            "account_number": "123456",
            "currency": "NA"
        },
        "associated_text": "Additionally, we have adjusted the fee structure for your account. The new fee is $200 effective immediately.",
        "explanation": "The email mentions an adjustment to the fee structure, which aligns with the 'Adjustment' category. The confidence score is high (0.85) because the email clearly describes a fee adjustment."
    }
]

"""
# Send the prompt to the model

'''
response = ollama.generate(
    model="0xroyce/plutus:latest",  # Specify the model
    prompt=prompt,           # Provide the prompt
    format="json"            # Request output in JSON format
)
'''

response = ollama.chat(model="0xroyce/plutus:latest", messages=[{'role': 'user','content': prompt}])
print(response['message']['content'])

print(response)

# Parse the response
'''
try:
    output = json.loads(response["response"])  # Parse the JSON output
    print(json.dumps(output, indent=4))       # Pretty-print the JSON
except json.JSONDecodeError:
    print("Error: The model's response is not valid JSON.")
    print("Raw Response:", response["response"])
'''
