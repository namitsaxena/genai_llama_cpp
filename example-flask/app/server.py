import json

from llama_cpp import Llama
from flask import Flask, request, jsonify
import os.path

# Put the location of to the GGUF model that you've download from HuggingFace here
model_name = "tinyllama-1.1b-chat-v0.3.Q2_K.gguf"
model_dir = "../../example-basic/models"


def get_path():
    model_path = None
    if os.path.isfile(model_name):
        model_path = model_name
    elif os.path.isfile(model_dir + '/' + model_name):
        model_path = model_dir + '/' + model_name
    else:
        raise Exception(f"Failed to find the model file for model: {model_name}")
    return model_path


model_path = get_path()
print(f"model file found at {model_path}")

# Create a Flask object
app = Flask("Llama server")
model = None


def query_llm(system_message, user_message, max_tokens):
    global model
    # Prompt creation
    prompt = f"""<s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {user_message} [/INST]"""

    # Create the model if it was not previously created
    if model is None:
        model = Llama(model_path=model_path)

    # Run the model
    output = model(prompt, max_tokens=max_tokens, echo=True)
    return output


@app.route('/llama', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        # Check if the required fields are present in the JSON data
        if 'system_message' in data and 'user_message' in data and 'max_tokens' in data:
            system_message = data['system_message']
            user_message = data['user_message']
            max_tokens = int(data['max_tokens'])
            output = query_llm(system_message, user_message, max_tokens)
            return jsonify(output)
        else:
            return jsonify({"error": "Missing required parameters"}), 400

    except Exception as e:
        return jsonify({"Error": str(e)}), 500


@app.route('/', methods=['GET'])
def generate_html_response():
    try:
        system_message = request.args.get('system_message', default="You are a helpful assistant")
        user_message = request.args.get('user_message', default="Generate a list of 5 funny dog names")
        max_tokens = request.args.get('max_tokens', type=int, default=500)

        if user_message and system_message:
            output = query_llm(system_message, user_message, max_tokens)
            output = output["choices"][0]['text']
            output = json.dumps(output)
            output = output.replace('<s>','<p/>')
            output = output.replace('\\n', '<br/>')
            return output
        else:
            return jsonify({"error": "Missing required parameters"}), 400

    except Exception as e:
        return jsonify({"Error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
