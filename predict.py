# C:\GrainProject\predict.py
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
def hard_sigmoid(x):
    return tf.keras.layers.ReLU(6.)(x + 3.) / 6.

def hard_swish(x):
    return x * hard_sigmoid(x)

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

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    print(f"Predicted Class: {predicted_class} ({confidence:.2f}%)")

    return predicted_class, confidence


def generate_description_gemini(predicted_class, confidence):
    genai.configure(api_key=os.environ.get("GENAI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
     Le modèle CNN analyse des images de GRAINS DE BLÉ.

    Il existe 8 classes possibles :
    - NOR : Grain normal
    - F&S : the Fusarium & Shriveled
    - SD : The sprouted grain
    - MY : The moldy grain
    - AP : The grain attacked by pests
    - BN : The broken grain
    - BP : The black point grain
    - HD : The heated grain

    Le modèle a prédit :
    Classe = {predicted_class}
    Confiance = {confidence:.2f}%

    Donne :
    1. Caractéristiques principales
    2. directive a suivre 
    3. les causes

    Réponse claire,courte,3 lignes max structurée et professionnelle.
    """

    response = model.generate_content(prompt)

    return response.text

def save_result_to_txt(filename, predicted_class, confidence, description):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== RESULTAT ANALYSE GRAIN ===\n\n")
        f.write(f"Classe prédite : {predicted_class}\n")
        f.write(f"Confiance : {confidence:.2f}%\n\n")
        f.write("=== DESCRIPTION GEMINI ===\n\n")
        f.write(description)

    print(f"\n Résultat sauvegardé dans {filename}")
# Dans predict.py
def generate_professional_pdf(predicted_class, confidence, description, output_pdf_path=None):
    """
    Génère un PDF professionnel pour l'analyse.
    Si output_pdf_path est fourni, le PDF sera sauvegardé à cet endroit.
    Sinon, le nom par défaut sera utilisé.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import A4
    from reportlab.graphics.barcode import qr
    from reportlab.graphics.shapes import Drawing
    from datetime import datetime

    if output_pdf_path is None:
        output_pdf_path = f"Rapport_Analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=20
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2E7D32"),
        spaceAfter=10
    )
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=6
    )
    red_title_style = ParagraphStyle(
        'RedTitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.red,
        spaceAfter=8
    )

    # === Titre ===
    elements.append(Paragraph("RAPPORT PROFESSIONNEL D'ANALYSE DES GRAINS", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # === Tableau résumé ===
    elements.append(Paragraph("Résumé de Classification", subtitle_style))
    table_data = [
        ["Classe prédite", predicted_class],
        ["Niveau de confiance", f"{confidence:.2f}%"],
        ["Date d'analyse", datetime.now().strftime("%d/%m/%Y %H:%M:%S")]
    ]
    table = Table(table_data, colWidths=[2.5 * inch, 3 * inch])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.4 * inch))

    # === Description ===
    elements.append(Paragraph("DESCRIPTION & RECOMMANDATIONS", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    cleaned_text = description.replace("###", "").replace("**", "").replace("*", "").replace("--", "")
    for line in cleaned_text.split("\n"):
        line = line.strip()
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(line, red_title_style))
        elif line != "":
            elements.append(Paragraph(line, normal_style))

    elements.append(Spacer(1, 0.5 * inch))

    # === QR Code ===
    qr_data = f"Classe: {predicted_class} | Confiance: {confidence:.2f}%"
    qr_code = qr.QrCodeWidget(qr_data)
    bounds = qr_code.getBounds()
    size = 120
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    d = Drawing(size, size, transform=[size/width, 0, 0, size/height, 0, 0])
    d.add(qr_code)
    elements.append(d)

    doc.build(elements)
    print(f"\n✅ PDF professionnel généré : {output_pdf_path}")


if __name__ == "__main__":

    image_path = "gg.png"

    predicted_class, confidence = predict_image(image_path)

    description = generate_description_gemini(
        predicted_class,
        confidence
    )

    print("\n=== RÉSULTAT FINAL ===")
    print(f"Classe : {predicted_class} ({confidence:.2f}%)")
    print("\nDescription :\n")
    print(description)

    save_result_to_txt("resultat_analyse.txt",
                       predicted_class,
                       confidence,
                       description)

    # ✅ Génération PDF professionnel
    generate_professional_pdf(
        predicted_class,
        confidence,
        description
    )


