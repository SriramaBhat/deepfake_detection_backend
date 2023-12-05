import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import flask_cors
import pickle
# import logging

# logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger('HELLO WORLD')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'ogg'])
TEXT_EXTENSIONS = set(["txt", "pdf", "docx"])
IMAGE_EXTENSIONS = set(["png", "jpeg", "jpg"])
AUDIO_EXTENSIONS = set(["mp3", "wav", "ogg"])

app = Flask(__name__)
flask_cors.CORS(app)

@app.route('/predict', methods=['POST'])
def fileUpload():
    file = request.form["file"]
    filename = request.form["name"]
    extension = filename.split(".")[1].strip()
    destination="/upload_files".join([filename])
    file.save(destination)
    if extension not in ALLOWED_EXTENSIONS:
      data = {
        "status_code": 400,
        "probability": -1,
        "message": "File extension not supported"
      }
      return jsonify(data)
    else:
      model_path = "/models"
      if extension in TEXT_EXTENSIONS:
        model_path += "/text_model.pkl"
      elif extension in IMAGE_EXTENSIONS:
        model_path += "/image_model.pkl"
      else:
        model_path += "/audio_model.pkl"
      model = pickle.load(model_path)
      probability = model.predict(destination)
      data = {
        "status_code": 201,
        "probability": probability,
        "message": "success"
      }
      return jsonify(data)

if __name__ == "__main__":
    app.run()

