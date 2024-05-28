import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
#cnn_model = model_from_json(loaded_model_json)
# load weights into new model
#cnn_model.load_weights("model.weights.h5")
# Load model
cnn_model = load_model('model.h5')
cnn_model.make_predict_function()

cnn1_model = load_model('model1.h5')
cnn1_model.make_predict_function()

IMAGE_SIZE = 150

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predict & classify image
# Predict & classify image
def classify_damage(cnn_model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    prob = cnn_model.predict(preprocessed_image)[0]
    print(prob)

    # Get the index of the maximum probability
    predicted_label_index = np.argmax(prob)

    # Mapping index to label name
    label_names = ['minor','moderate','severe','NORMAL']
    # Replace with your actual label names

    label = label_names[predicted_label_index]

    classified_prob = prob[predicted_label_index]

    return label, classified_prob

def classify_brand(cnn1_model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    prob = cnn1_model.predict(preprocessed_image)[0]
    print(prob)

    # Get the index of the maximum probability
    predicted_label_index = np.argmax(prob)

    # Mapping index to label name
    label_names = ['AUDI','BMW','TOYOTA']
    # Replace with your actual label names

    label = label_names[predicted_label_index]

    classified_prob = prob[predicted_label_index]

    return label, classified_prob



# home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify_damage", methods=["POST", "GET"])
def upload_damage_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify_damage(cnn_model, upload_image_path)

        prob = round((prob * 100), 2)

    return render_template(
        "classify_damage.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/classify_brand", methods=["POST", "GET"])
def upload_brand_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify_brand(cnn1_model, upload_image_path)

        prob = round((prob * 100), 2)

    return render_template(
        "classify_brand.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify_damage/<filename>")
def send_damage_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/classify_brand/<filename>")
def send_brand_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":

    app.run()

