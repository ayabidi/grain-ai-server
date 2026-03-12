# C:\GrainProject\server.py
from flask import Flask, request, jsonify, send_from_directory
from predict import predict_image, generate_description_gemini, generate_professional_pdf, save_result_to_txt
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PDF_FOLDER = os.path.join(UPLOAD_FOLDER, "pdfs")
os.makedirs(PDF_FOLDER, exist_ok=True)

# === Servir les PDF générés via HTTP ===
@app.route("/pdf/<filename>", methods=["GET"])
def serve_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename)

# === Analyse d'image ===
@app.route("/analyze", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 1️⃣ Prédiction
    predicted_class, confidence = predict_image(file_path)

    # 2️⃣ Description avec nettoyage
    description = generate_description_gemini(predicted_class, confidence)
    cleaned_description = description.replace("###", "") \
                                     .replace("**", "") \
                                     .replace("*", "") \
                                     .replace("--", "")

    # 3️⃣ Sauvegarde texte
    txt_path = os.path.join(UPLOAD_FOLDER, "result.txt")
    save_result_to_txt(txt_path, predicted_class, confidence, cleaned_description)

    # 4️⃣ Génération PDF
    pdf_filename = f"Rapport_Analyse.pdf"  # fixe ou timestamp si tu veux versions multiples
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    generate_professional_pdf(predicted_class, confidence, cleaned_description, pdf_path)

    # 5️⃣ Retour JSON pour Flutter
    # Remplace l'IP par celle de ton PC sur le réseau local
    pdf_url = f"{request.url_root}pdf/{pdf_filename}"


    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "description": cleaned_description,
        "pdf_url": pdf_url,   # URL HTTP pour mobile
        "txt_path": txt_path
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render définit ce PORT automatiquement
    app.run(host="0.0.0.0", port=port)
