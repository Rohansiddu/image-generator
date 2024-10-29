from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)
load_dotenv()

# Azure Cognitive Services configuration
cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
cog_key = os.getenv('COG_SERVICE_KEY')
credentials = CognitiveServicesCredentials(cog_key)
vision_client = ComputerVisionClient(cog_endpoint, credentials)

# OpenAI configuration
api_base = os.getenv("AZURE_OAI_ENDPOINT")
api_key = os.getenv("AZURE_OAI_KEY")
api_version = '2024-02-15-preview'

def AnalyzeImage(image_stream):
    """
    Analyzes the image to gather tags and description using Azure Computer Vision.
    """
    print('Analyzing image')
    analysis = vision_client.analyze_image_in_stream(
        image_stream,
        visual_features=[VisualFeatureTypes.tags, VisualFeatureTypes.description]
    )

    # Extract tags and description
    tags = [tag.name for tag in analysis.tags]
    description = analysis.description.captions[0].text if analysis.description.captions else "a cityscape"
    return tags, description

def create_future_city_prompt(description, tags):
    """
    Creates a prompt for generating a futuristic city image based on image description and tags.
    """
    future_features = ", ".join([
        "taller skyscrapers", "advanced architecture", "more green spaces",
        "futuristic transportation systems", "eco-friendly buildings"
    ])
    
    prompt = (
        f"A futuristic version of a city that includes {description}. "
        f"The city has {future_features}. Key features from the original city include: {', '.join(tags)}."
    )
    return prompt

def generate_future_city_image(prompt):
    """
    Calls the DALL-E model to generate a future city image based on the generated prompt.
    """
    url = f"{api_base}openai/deployments/dep-03/images/generations?api-version={api_version}"
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    # Print the entire response for debugging
    print("API Response:", response.json())
    
    # Check if 'data' is in the response
    if 'data' in response.json():
        revised_prompt = response.json()['data'][0]['revised_prompt']
        image_url = response.json()['data'][0]['url']
    else:
        raise ValueError("Unexpected response format: 'data' key not found.")
    
    return revised_prompt, image_url

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    try:
        # Analyze the image using your existing function
        tags, description = AnalyzeImage(image_file.stream)
        
        # Generate prompt for the future city
        prompt = create_future_city_prompt(description, tags)
        
        # Generate the future city image using your existing function
        revised_prompt, image_url = generate_future_city_image(prompt)
        
        # Prepare the result to return
        result = {
            'description': description,
            'tags': tags,
            'prompt': revised_prompt,
            'imageUrl': image_url
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
