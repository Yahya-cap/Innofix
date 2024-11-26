import base64
from gen_ai_hub.proxy.native.openai import chat
import requests
from flask import Flask, request, jsonify
import hdbcli
from hdbcli import dbapi
import hana_ml
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.native.openai import embeddings
from langchain.prompts import PromptTemplate
import tiktoken
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.native.openai import chat
import os

# Database connection setup
# Import necessary modules, and define a function to query an LLM with a formatted prompt and vector-based context
# Create a prompt template
promptTemplate_fstring = """
You are an SAP HANA Cloud expert.
You are provided multiple context items that are related to the prompt you have to answer.
Use the following pieces of context to answer the question at the end.
Context:
{context}
Question:
{query}
"""

promptTemplate = PromptTemplate.from_template(promptTemplate_fstring)

proxy_client = get_proxy_client('gen-ai-hub')  # for an AI Core proxy
 
port = int(os.environ.get('PORT', 3000))
app = Flask(__name__)
cc = dbapi.connect(
    address='794d7a51-4a96-4b97-a476-dacb3a0440b4.hna2.prod-eu10.hanacloud.ondemand.com',
    port=443,
    user='CC4AFC2F627D4914A44DD6C261D7FF8D_ERSTH1BXP245L82B8489DZI58_RT',
    password='Ya2Eyj6fRxkXOV7lIA2no79-1YmCUV0rF.gDA71X-Kgh..Q6RnS2Ll8xTxpr4A6BLCVg418g1bCishkKMOzEy0xnT_qIzp4lexALzL-S230emqkBsx-N.kUai5.y2R2q',
    encrypt=True
)
cursor = cc.cursor()

print("Database connection established.")

# Function to get embedding for a query
def get_embedding(input, model="text-embedding-ada-002") -> str:
    response = embeddings.create(
        model_name=model,
        input=input
    )
    return response.data[0].embedding

# Vector search function
def run_vector_search(query: str, metric="COSINE_SIMILARITY", k=3):
    """Recherche dans DEFECTS_TABLE_FINAL pour les enregistrements les plus similaires."""
    if metric == 'L2DISTANCE':
        sort = 'ASC'
    else:
        sort = 'DESC'
   
    query_vector = get_embedding(query)
   
    sql = '''
    SELECT TOP {k} "NC_NUMBER", "DESCRIPTION", "VECTOR_STR",
        {metric}("VECTOR_STR", TO_REAL_VECTOR('{qv}')) AS similarity_score
    FROM "DEFECTS_TABLE_FINAL"
    ORDER BY similarity_score {sort}
    '''.format(k=k, metric=metric, qv=query_vector, sort=sort)
   
    cursor.execute(sql)
    results = cursor.fetchall()
   
    return [(row[0], row[1], row[2], row[3]) for row in results]



# Retrieve default code for NC_NUMBER
def get_default_code(nc_number):
    sql = '''
    SELECT "DEFAULT_CODE"
    FROM "DEFECTS_TABLE_FINAL"
    WHERE "NC_NUMBER" = ?
    '''
    cursor.execute(sql, (nc_number,))
    result = cursor.fetchone()
    return result[0] if result else None

# Retrieve repair procedure for DEFAULT_CODE
def get_repair_procedure(default_code):
    sql = '''
    SELECT "REPAIR_PROCEDURE"
    FROM "REPAIR_TABLE"
    WHERE "CODE" = ?
    '''
    cursor.execute(sql, (default_code,))
    result = cursor.fetchone()
    return result[0] if result else "No repair procedure found."


 
def generate_text_with_mistral(description, context):
    """
    Generates a repair procedure based on a user's description of a defect
    and a given repair context.
    
    Parameters:
        description (str): A description of the defect provided by the user.
        context (str): The initial repair procedure or context to work from.
    
    Returns:
        str: A professionally reformulated repair procedure.
    """
    messages = [
        {"role": "system", "content": "You are an expert in aircraft maintenance and repair."},
        {"role": "user", "content": f"A user has described a defect as follows: '{description}'. "
                                     f"Given this defect, suggest a repair procedure. "
                                     f"Here is some initial repair context: {context}. "
                                     "Please provide a clear and professional repair procedure."}
    ]

    response = chat.completions.create(
        model_name="mistralai--mistral-large-instruct",
        messages=messages
    )

    return response.to_dict()["choices"][0]["message"]["content"]



# New endpoint to process image and description
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Parse request data
        data = request.get_json()
        if not data or 'image' not in data or 'description' not in data:
            return jsonify({"error": "Both 'image' and 'description' must be provided"}), 400
        
        # Extract image and description
        encoded_image = data['image']
        description = data['description']
        
        # Decode the Base64 image (optional: save or process it)
        try:
            pass
            #image_data = base64.b64decode(encoded_image)
        except Exception as e:
            return jsonify({"error": f"Invalid Base64 image: {str(e)}"}), 400


        # Perform vector search
        similar_ncs = run_vector_search(description, k=3)
        if not similar_ncs:
            return jsonify({"error": "No similar NCs found"}), 404

        # Extract the most similar NC's default code
        first_nc_number = similar_ncs[0][0]
        default_code = get_default_code(first_nc_number)
        if not default_code:
            return jsonify({"error": f"No default code found for NC {first_nc_number}"}), 404

        # Retrieve the repair procedure for the default code
        repair_procedure = get_repair_procedure(default_code)
        suggested_procedure = generate_text_with_mistral(repair_procedure,description)
        # Construct the response
        response = {
            "suggested repair procedure": suggested_procedure,
            "similar_NCs": [
                {"NC_NUMBER": nc[0], "similarity_score": int(nc[3] * 100)} for nc in similar_ncs
            ],  # NC_NUMBER and similarity score as a percentage
            "repair_procedure": repair_procedure
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


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
    app.run(host='0.0.0.0', port=port, debug=True)