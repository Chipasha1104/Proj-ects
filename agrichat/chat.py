import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure generative AI with API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set generation and safety configurations
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model with specified settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=(
        "Respond as a knowledgeable and friendly agricultural expert, providing clear and concise explanations, "
        "suggesting specific solutions, and reassuring the user with encouragement. Avoid technical jargon by using "
        "simple language, tailor your response to the specific question, and offer hopeful solutions. "
        "Do not answer anything outside of agriculture."
    ),
)

# Create a new Flask app
app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Render a simple HTML interface

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    
    # Start a new chat session
    chat_session = model.start_chat(history=[])
    
    # Get the model's response
    response = chat_session.send_message(user_input)
    model_response = response.text
    
    return jsonify({"response": model_response})

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the request contains an image file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Secure the file and save it
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Placeholder for disease detection function
        # Call your AI model or detection function here using file_path
        # Example: result = detect_disease(file_path)
        
        return jsonify({"result": "Disease detected and solution provided"}), 200
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
