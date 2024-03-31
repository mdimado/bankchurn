from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os

app = Flask(__name__)

# Configure YOLO model
model = YOLO("/Users/mohammedimaduddin/Desktop/bankchurn/bankchurn/backend/best.pt")

# Define allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return render_template('index.html', error='File type not allowed')

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Perform prediction using YOLO model
    results = model.predict(file_path)

    # Extract relevant information from the prediction result
    predictions = []
    for result in results:
        boxes = result.pred[0].tolist()  # Convert bounding boxes to list for easy rendering
        class_names = result.names
        predictions.extend([{'class': class_names[int(box[5])], 'confidence': box[4]} for box in boxes])

    # Delete the uploaded file after prediction
    os.remove(file_path)

    return render_template('index.html', predictions=predictions)


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static'  # Change this to your desired upload folder
    app.run(debug=True)
