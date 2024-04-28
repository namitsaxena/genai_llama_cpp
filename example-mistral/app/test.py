from llama_cpp import Llama
import os.path

# Put the location of to the GGUF model that you've download from HuggingFace here
model_name = "tinyllama-1.1b-chat-v0.3.Q2_K.gguf"
model_dir = "../models"

# determine path for the model file, both
# for running in Docker(same dir) and
# for running in IDE (under separate dir)
model_path = None
if os.path.isfile(model_name):
    model_path = model_name
elif os.path.isfile(model_dir + '/' + model_name):
    model_path = model_dir + '/' + model_name
else:
    print(f"Failed to find the model file for model: {model_name}")
    exit(1)

print(f"model file found at {model_path}")

# Create a llama model
model = Llama(model_path)

# Prompt creation
# system_message = "You are a helpful software developer"
# user_message = "Hello! Can you write a 300 word abstract for a research paper I need to write about the impact of AI on society?"

system_message = "You are a chef"
user_message = "Give me an easy and detailed recipe for making pancakes."

prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""

# Model parameters
max_tokens = 500

# Run the model
output = model(prompt, max_tokens=max_tokens, echo=True)

# Print the model output
print(output)