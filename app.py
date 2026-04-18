import os
import csv
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image

# ================= SETUP =================
app = Flask(__name__)
CORS(app)

# ================= MODEL LOAD =================
tf.keras.backend.clear_session()

MODEL_PATH = "food_model.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading error:", e)
    model = None

# ================= LABELS =================
label = ['apple pie','baby back ribs','baklava','beef carpaccio','beef tartare',
'beet salad','beignets','bibimbap','bread pudding','breakfast burrito',
'bruschetta','caesar salad','cannoli','caprese salad','carrot cake','ceviche',
'cheese plate','cheesecake','chicken curry','chicken quesadilla','chicken wings',
'chocolate cake','chocolate mousse','churros','clam chowder','club sandwich',
'crab cakes','creme brulee','croque madame','cup cakes','deviled eggs','donuts',
'dumplings','edamame','eggs benedict','escargots','falafel','filet mignon',
'fish and_chips','foie gras','french fries','french onion soup','french toast',
'fried calamari','fried rice','frozen yogurt','garlic bread','gnocchi',
'greek salad','grilled cheese sandwich','grilled salmon','guacamole','gyoza',
'hamburger','hot and sour soup','hot dog','huevos rancheros','hummus',
'ice cream','lasagna','lobster bisque','lobster roll sandwich',
'macaroni and cheese','macarons','miso soup','mussels','nachos','omelette',
'onion rings','oysters','pad thai','paella','pancakes','panna cotta',
'peking duck','pho','pizza','pork chop','poutine','prime rib',
'pulled pork sandwich','ramen','ravioli','red velvet cake','risotto',
'samosa','sashimi','scallops','seaweed salad','shrimp and grits',
'spaghetti bolognese','spaghetti carbonara','spring rolls','steak',
'strawberry shortcake','sushi','tacos','octopus balls','tiramisu',
'tuna tartare','waffles']

# ================= NUTRITION =================
nutrition_table = {}

try:
    with open("nutrition101.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            name = row[1].strip()
            nutrition_table[name] = [
                {"name": "protein", "value": float(row[2])},
                {"name": "calcium", "value": float(row[3])},
                {"name": "fat", "value": float(row[4])},
                {"name": "carbohydrates", "value": float(row[5])},
                {"name": "vitamins", "value": float(row[6])},
            ]

    print("Nutrition data loaded successfully")

except Exception as e:
    print("Nutrition CSV error:", e)

# ================= LOGIC =================
def carb_status(value):
    if value <= 10:
        return "green", "Safe"
    elif value <= 25:
        return "yellow", "Moderate"
    else:
        return "red", "Avoid"

# ================= ROUTE =================
@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        if "img" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["img"]

        # preprocess image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        # prediction
        pred = model.predict(img)

        top = pred.argsort()[0][-3:]
        best_index = top[2]
        best = label[best_index]

        # nutrition lookup
        nutrition = nutrition_table.get(best, [
            {"name": "protein", "value": 0},
            {"name": "calcium", "value": 0},
            {"name": "fat", "value": 0},
            {"name": "carbohydrates", "value": 0},
            {"name": "vitamins", "value": 0},
        ])

        carbs = nutrition[3]["value"] if len(nutrition) > 3 else 0
        color, status = carb_status(carbs)

        confidence = float(pred[0][best_index]) * 100

        return jsonify({
            "success": True,
            "food_name": best,
            "confidence": round(confidence, 2),
            "carbs": carbs,
            "carb_status": status,
            "nutrition": nutrition,
            "food_link": "https://world.openfoodfacts.org/cgi/search.pl?search_terms=" + best
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ================= RUN (HUGGING FACE SAFE) =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)