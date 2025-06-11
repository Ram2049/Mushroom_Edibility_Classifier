from flask import Flask,request, jsonify,render_template
import os
from flask_cors import CORS,cross_origin
from werkzeug.utils import secure_filename
from Mushroom_edibility_classifier.utils.common import decodeImage
from Mushroom_edibility_classifier.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        # Initialize pipeline without filename
        self.classifier = PredictionPipeline()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')
# Add allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed file types are: png, jpg, jpeg, webp'}), 400
    
    # Securely save the file
    filename = secure_filename(file.filename)
    temp_path = os.path.join('temp_uploads', filename)
    os.makedirs('temp_uploads', exist_ok=True)
    file.save(temp_path)
    
    try:
        result = clApp.classifier.predict(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ... (rest of your app.py remains the same)    
if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    app.run(host='0.0.0.0', port=80) #for AZURE
    