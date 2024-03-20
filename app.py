from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import transformers
import tensorflow_text as text
import os

UPLOAD_FOLDER = "./static"
ALLOWED_EXTENSIONS = ["txt", "docx", "pdf", "jpeg", "jpg", "png", "mp3", "wav", "ogg"]

app = Flask(__name__)
cors = CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_and_predict_text(text):
  text_model = tf.keras.models.load_model("./models/bert_model (1)")
  return text_model

@app.route("/predict", methods=["GET"])
def predict_deepfake():
  model = load_and_predict_text("txt")
  return model.summary()
