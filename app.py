import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

UPLOAD_FOLDER = "./static"
ALLOWED_EXTENSIONS = ["txt", "docx", "pdf", "jpeg", "jpg", "png", "mp3", "wav", "ogg"]

app = Flask(__name__)
cors = CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/predict", methods=["POST"])
def predict_deepfake():
  return 0
