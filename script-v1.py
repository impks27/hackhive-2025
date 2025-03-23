import ollama

# Define file paths for different sections
objective_file = "resources/objective.txt"
categories_file = "resources/categories.txt"
email_to_classify_file = "resources/email_to_classify.txt"
instructions_file = "resources/instructions.txt"

# Function to read content from a file
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
        return ""

# Read each section
objective = read_file(objective_file)
categories = read_file(categories_file)
email_to_classify = read_file(email_to_classify_file)
instructions = read_file(instructions_file)

# Combine all sections into the final prompt
prompt = f"{objective}\n\n{categories}\n\n{email_to_classify}\n\n{instructions}"

print("🚀 Sending the prompt to the AI model... Stand by for classification!")
#print(prompt)

# Send the prompt to the model
response = ollama.chat(model="0xroyce/plutus:latest", messages=[{'role': 'user', 'content': prompt}])

# Print the response content
print("📊 Data processed! Here’s the classified breakdown:")
print(response['message']['content'])
