import base64
from flask import Flask, request, jsonify
from gen_ai_hub.proxy.native.openai import chat
import requests

app = Flask(__name__)

def create_image_prompt(image_url: str, text_prompt: str) -> list:
    """Create a prompt for the model with both image URL and text description."""
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Yahya-cap/test-images/refs/heads/main/test.png"}}
        ]
    }]

def get_model_response(model_name: str, messages: list) -> str:
    """Send messages to the model and return its response."""
    response = chat.completions.create(model_name=model_name, messages=messages)
    return response.to_dict()["choices"][0]["message"]["content"]

def process_image_and_description(encoded_image: str, description: str, model_name: str) -> str:
    """Process the base64-encoded image and description, then interact with the AI model."""
    try:
        # Optional: Implement image processing logic here if necessary
        # image_data = base64.b64decode(encoded_image)  # Decode image if needed

        # Create the prompt to send to the model
        image_url = "dummy_image_url_for_processing"  # Placeholder, as no image processing is done here
        text_prompt = f"Describe this image and identify defects. Description: {description}"
        messages = create_image_prompt(image_url, text_prompt)
        
        # Get the response from the model
        return get_model_response(model_name, messages)

    except Exception as e:
        return f"Error processing image: {str(e)}"

@app.route('/Description', methods=['POST'])
def handle_request():
    """Handle the incoming request to process image and description."""
    try:
        data = request.get_json()

        # Ensure both image and description are provided
        if not data or 'image' not in data or 'description' not in data:
            return jsonify({"error": "Both image and description are required"}), 400

        encoded_image = data['image']
        description = data['description']
        
        # Call the image processing function
        model_name = "gpt-4o"  # Replace with the correct model name
        result = process_image_and_description(encoded_image, description, model_name)
        
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
