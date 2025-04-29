from flask import Flask, request, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import cv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

embeddings_np = np.load("image_embeddings.npy")
nn_model = joblib.load("NN_model.pkl")
df = pd.read_csv("filtered_products.csv")
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    embedding = model(img)
    return embedding.numpy().flatten()

html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Herfety Search</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .card { border: 1px solid #ccc; border-radius: 8px; padding: 10px; text-align: center; }
        .card img { max-width: 100%; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>🔍 بحث مرئي عن المنتجات</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">بحث</button>
    </form>
    {% if uploaded_image %}
        <h3>✔️ الصورة التي تم رفعها:</h3>
        <img src="{{ uploaded_image }}" width="300">
    {% endif %}
    {% if results %}
        <h2>النتائج:</h2>
        <div class="results">
        {% for item in results %}
            <div class="card">
                <img src="{{ item.image_url }}">
                <h3>{{ item.name }}</h3>
                <p>{{ item.price }} {{ item.currency }}</p>
            </div>
        {% endfor %}
        </div>
    {% endif %}
</body>
</html>"""

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    uploaded_image = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            uploaded_image = '/' + img_path

            query_embedding = extract_embedding(img_path)
            distances, indices = nn_model.kneighbors([query_embedding])

            for idx in indices[0]:
                results.append({
                    'name': df.iloc[idx]['name'],
                    'price': df.iloc[idx]['price'],
                    'currency': df.iloc[idx]['currency'],
                    'image_url': df.iloc[idx]['images.1']
                })

    return render_template_string(html_template, results=results, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run()