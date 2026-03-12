from flask import Flask, request, jsonify
from predict import predict_image, generate_description_gemini
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    predicted_class, confidence = predict_image(path)

    description = generate_description_gemini(
        predicted_class,
        confidence
    )

    return jsonify({
        "class": predicted_class,
        "confidence": confidence,
        "description": description
    })


if __name__ == "__main__":
    app.run()