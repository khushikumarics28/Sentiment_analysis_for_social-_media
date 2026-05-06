from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    prediction = model.predict([text])[0]

    print("Input:", text)
    print("Prediction:", prediction)

    return jsonify({
        "label": prediction
    })

if __name__ == '__main__':
    app.run(debug=True)