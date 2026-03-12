import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from datetime import datetime
import threading

# === Fonctions d'activation personnalisées ===
def hard_sigmoid(x):
    return tf.keras.layers.ReLU(6.)(x + 3.) / 6.

def hard_swish(x):
    return x * hard_sigmoid(x)

# === Chargement du modèle TensorFlow ===
print("Loading AI model...")
model = load_model(
    "modelv34-2.h5",
    custom_objects={"hard_swish": hard_swish},
    compile=False
)
print("Model loaded successfully")

class_names = [
    "Grain normal",
    "the Fusarium & Shriveled",
    "The sprouted grain",
    "The moldy grain",
    "The grain attacked by pests",
    "The broken grain",
    "The black point grain",
    "The heated grain"
]

# === Prédiction d'image ===
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100
    return predicted_class, confidence

# === Génération texte avec fallback ===
def generate_description_safe(predicted_class, confidence):
    try:
        genai.configure(api_key=os.environ.get("GENAI_API_KEY"))
        g_model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Classe prédite = {predicted_class}, Confiance = {confidence:.2f}%.
        Donne 1. Caractéristiques, 2. Directive, 3. Causes, 3 lignes max.
        """
        response = g_model.generate_content(prompt)
        return response.text
    except:
        return "Description non disponible pour le moment."

# === Sauvegarde TXT ===
def save_result_to_txt(filename, predicted_class, confidence, description):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Classe : {predicted_class}\nConfiance : {confidence:.2f}%\n\n{description}\n")

# === Génération PDF ===
def generate_professional_pdf(predicted_class, confidence, description, output_pdf_path=None):
    if output_pdf_path is None:
        output_pdf_path = f"Rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"Classe : {predicted_class} ({confidence:.2f}%)", styles['Heading2']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(description, styles['Normal']))
    doc.build(elements)

# === Thread pour TXT + PDF ===
def generate_results_async(predicted_class, confidence, image_path):
    description = generate_description_safe(predicted_class, confidence)
    save_result_to_txt(f"{image_path}_result.txt", predicted_class, confidence, description)
    generate_professional_pdf(predicted_class, confidence, description, f"{image_path}_report.pdf")

# === Exemple exécution locale ===
if __name__ == "__main__":
    image_path = "gg.png"
    predicted_class, confidence = predict_image(image_path)
    print(f"Classe : {predicted_class} ({confidence:.2f}%)")
    threading.Thread(target=generate_results_async, args=(predicted_class, confidence, image_path)).start()
