from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. SETUP GEMINI
# ==========================================
# 🛑 PASTE YOUR API KEY HERE
MY_API_KEY = "Here"
genai.configure(api_key=MY_API_KEY)

# Auto-detect best model
print("⚙️ Configuring Gemini Model...")
priority_list = ["models/gemini-flash-latest", "models/gemini-pro-latest", "models/gemini-pro"]
selected_model = "models/gemini-pro"
try:
    avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    for p in priority_list:
        if p in avail:
            selected_model = p
            break
except:
    pass
print(f"✅ Using Gemini Model: {selected_model}")
gem_model = genai.GenerativeModel(selected_model)

# ==========================================
# 2. LOAD ALL 5 EXISTING MODELS
# ==========================================
base_path = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_path, "ml_models")
models = {}

disease_names = ["heart", "diabetes", "liver", "kidney", "cancer"]

print("\n--- LOADING MODELS ---")
for name in disease_names:
    path = os.path.join(model_dir, f"{name}.pkl")
    if os.path.exists(path):
        try:
            # Try loading as (model, scaler) tuple
            loaded_data = pickle.load(open(path, "rb"))
            if isinstance(loaded_data, tuple) or isinstance(loaded_data, list):
                models[name] = loaded_data # It's a tuple (model, scaler)
            else:
                models[name] = (loaded_data, None) # It's just the model, no scaler
            print(f"✅ Loaded: {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    else:
        print(f"⚠️ Warning: {name}.pkl not found in ml_models folder")

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
def run_prediction(disease, data):
    if disease not in models:
        return {"error": f"Model for {disease} is not loaded."}
    
    try:
        model_data = models[disease]
        model = model_data[0]
        scaler = model_data[1]

        # Auto-clean input: Convert strings (Male/Female) to numbers (1/0)
        clean_data = []
        for x in data:
            if isinstance(x, str):
                val = 1 if x.lower() in ['male', 'm', 'yes', 'true'] else 0
                clean_data.append(val)
            else:
                clean_data.append(float(x))

        # Scale if a scaler exists
        if scaler:
            final_input = scaler.transform([clean_data])
        else:
            final_input = [clean_data]

        # Predict
        prediction = int(model.predict(final_input)[0])
        status = "Positive (High Risk)" if prediction == 1 else "Negative (Healthy)"
        
        # Ask Gemini for a summary
        prompt = f"Act as a doctor. Patient checked for {disease}. Result: {status}. Explain briefly and give a recommendation."
        try:
            summary = gem_model.generate_content(prompt).text
        except:
            summary = "Consult a doctor."

        return {
            "prediction_code": prediction,
            "result_text": status,
            "gemini_summary": summary
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": str(e)}

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    return jsonify(run_prediction(disease, request.json.get("input")))

# ==========================================
# 4. CHAT ROUTE
# ==========================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    context = data.get("context", "")
    user_msg = data.get("message", "")
    
    prompt = f"""
    Act as a medical assistant named Medicure.
    Context from Report: {context}
    User Question: {user_msg}
    Keep answer short, polite, and strictly about health.
    """
    try:
        response = gem_model.generate_content(prompt).text
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"reply": "I am having trouble connecting. Please try again."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)