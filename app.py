import json
import tensorflow as tf
from tensorflow import keras
from flask import Flask
from flask import request, jsonify
from PIL import Image
import os

# To force inference using CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model definition

image_input = keras.Input(shape=(None, None, 3))

x = keras.layers.Resizing(
    height=224, width=224, interpolation="lanczos3", crop_to_aspect_ratio=False
)(image_input)

x = keras.layers.Rescaling(scale=1.0 / 255, offset=0.0)(x)

mobilenet = keras.applications.MobileNetV2(
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=image_input,
    classes=1000,
    classifier_activation="softmax",
)

model_output = mobilenet(x)

model = keras.Model(inputs=image_input, outputs=model_output)

# Function for inference


def inference(image: tf.Tensor):
    y = model(image).numpy()
    preds = keras.applications.imagenet_utils.decode_predictions(y, top=5)
    result = {i[1]: str(i[2]) for i in preds[0]}
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
    return result


# Defining simple Flask app and two endpoints for it
# One is a health-check on the root URL, while the other is the inference endpoint

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    result = {"outcome": "endpoint working successfully"}
    return jsonify(result)


@app.route("/inference", methods=["GET", "POST"])
def perform_inference():
    try:
        image = request.files["image"]
        state = 'image received.'
        pil_img = Image.open(image.stream)
        state = 'pil img created.'
        tensor = keras.preprocessing.image.img_to_array(pil_img)
        state = 'tensor done'
        tensor = tf.expand_dims(tensor, axis=0)
        state = 'expand dims done'
        result = inference(tensor)
        state = 'inference complete'
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "state": state})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
