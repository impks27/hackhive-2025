import ollama
import json

# Define the prompt
# Read the prompt from the file
with open("resources/prompt.txt", "r", encoding="utf-8") as file:
    prompt = file.read()
    
# Send the prompt to the model
response = ollama.chat(model="0xroyce/plutus:latest", messages=[{'role': 'user','content': prompt}])
print(response['message']['content'])

print(response)