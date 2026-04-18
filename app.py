import os
import csv
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from flask_cors import CORS
from PIL import Image 

load_dotenv()

# ================= SETUP =================
app = Flask(__name__)
CORS(app)

# ================= LOAD MODEL =================
tf.keras.backend.clear_session()
model = load_model("food_model.h5", compile=False)

# ================= FOOD LABELS =================
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

# ================= NUTRITION DATA =================
nutrition_table = {}

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

# ================= LOGIC =================
def carb_status(value):
    if value <= 10:
        return "green", "Safe"
    elif value <= 25:
        return "yellow", "Moderate"
    else:
        return "red", "Avoid"

# ================= API =================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    
    try:
        print(request.files)  
        
        if "img" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["img"]

       
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        pred = model.predict(img)

        top = pred.argsort()[0][-3:]
        best = label[top[2]]

        nutrition = nutrition_table.get(best, [
            {"name": "protein", "value": 0},
            {"name": "calcium", "value": 0},
            {"name": "fat", "value": 0},
            {"name": "carbohydrates", "value": 0},
            {"name": "vitamins", "value": 0},
        ])

        carbs = nutrition[3]["value"] if len(nutrition) > 3 else 0
        color, status = carb_status(carbs)

        return jsonify({
            "success": True,
            "food_name": best,
            "confidence": float(pred[0][top[2]] * 100),
            "carbs": carbs,
            "carb_status": status,
            "nutrition": nutrition,
            "food_link": "https://world.openfoodfacts.org/cgi/search.pl?search_terms=" + best
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)