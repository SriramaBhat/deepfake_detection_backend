from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
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

def convert_image(filename):
  img = Image.open(f"./static/images/{filename}")
  gray = 0
  if img.mode == "RGBA" or img.mode == "P": img = img.convert("RGB")
  if img.mode == "L" or len(img.getpixel((0, 0))) == 1: gray = 1
  name = filename.split(".")
  del name[-1]
  name = ".".join(name)
  img.save(f"./static/images/{name}.jpg", quality=75, optimize=True)
  if filename != f"{name}.jpg": os.remove(f"./static/images/{filename}")
  return {"filename": f"{name}.jpg", "gray": gray}

def preprocess_image(filename, isGray):
  img = tf.io.read_file(f"./static/images/{filename}")
  if isGray:
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.grayscale_to_rgb(img)
  else:
    img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, size=[224, 224])
  img = np.expand_dims(img, axis=0)
  return img

def load_and_predict_text(text):
  text_model = tf.keras.models.load_model("./models/bert_model (1)")
  text = np.expand_dims(text, axis=0)
  prediction = text_model.predict(text)
  prediction = prediction.tolist()[0][0]
  return prediction

def load_and_predict_image(filename):
  image_model = tf.keras.models.load_model("./models/NewnetV1_2")
  obj = convert_image(filename)
  img = preprocess_image(obj["filename"], obj["gray"])
  prediction = image_model.predict(img)
  prediction = prediction.tolist()[0][0]
  return prediction

@app.route("/predict", methods=["GET"])
def predict_deepfake():
  # prediction = load_and_predict_text(("Adding batch size to input data is straightforward in Python. If you\'re using TensorFlow or PyTorch, you can simply reshape your input data to include an additional dimension representing the batch size."))
  # prediction = load_and_predict_image("test2.png")
  return jsonify({"prediction": "Hello"})
