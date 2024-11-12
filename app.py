from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import easyocr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image  # Import for handling images
import io  # Import for byte stream handling
import numpy as np  # Import numpy for array handling

# Initialize the Flask app 
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load dataset and clean only the 'Prompt' column
df = pd.read_csv("Doj_data.csv", encoding='latin1')  # Use appropriate encoding
df.drop_duplicates(inplace=True)

# Replace NaN values with empty strings in both Prompt and Response columns
df['Prompt'] = df['Prompt'].fillna('')
df['Response'] = df['Response'].fillna('')

# Define stop words set for cleaning
stop_words = set(stopwords.words("english"))

# Clean text function (removes special characters except alphanumeric, keeps numbers)
def clean_text(text):
    if not isinstance(text, str):  # Check if text is a string
        return ''  # Return empty string for non-string inputs
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(word for word in text.split() if word not in stop_words)

# Apply cleaning only to the 'Prompt' column for similarity matching
df['Cleaned_Prompt'] = df['Prompt'].apply(clean_text)

# Load pre-trained SentenceTransformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all cleaned prompts in the dataset
query_embeddings = model.encode(df['Cleaned_Prompt'].tolist())

# Initialize easyocr Reader
reader = easyocr.Reader(['en'])

def get_response(user_query):
    # Clean and encode the user query
    cleaned_query = clean_text(user_query)
    user_embedding = model.encode([cleaned_query])
    
    # Calculate semantic similarity with cosine similarity
    similarities = cosine_similarity(user_embedding, query_embeddings)
    best_match_index = similarities.argmax()
    
    # Check if similarity is above a certain threshold
    if similarities[0][best_match_index] > 0.5:  # Set threshold for accuracy
        return df.iloc[best_match_index]['Response']  # Return the original response
    else:
        return "I'm sorry, I couldn't find relevant information related to that query."

# Function to extract text from an uploaded image
def extract_text_from_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        result = reader.readtext(np.array(image))
        text = ' '.join([res[1] for res in result])
        return text
    except Exception as e:
        return str(e)

# Define the index route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the chat route to handle user queries
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['query']
    
    try:
        # Get response from the chatbot function
        response = get_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

# Define the file upload route for image text extraction
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Read the uploaded image or text file
        image_data = file.read()
        extracted_text = extract_text_from_image(image_data)

        if extracted_text:
            response = get_response(extracted_text)  # Using get_response for consistency
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'Could not extract text from the image.'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
