from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to your Zaniar API!"

@app.route('/api/', methods=['GET'])
def get_data():
    data = {"message": "This API only supports POST method!"}
    return jsonify(data)

@app.route('/api/', methods=['POST'])
def post_data():
    received_data = request.json
    return jsonify({"received_data": received_data}), 201

def data_clean(data):
    return None

def predict(data):
    clf2 = joblib.load("model.pkl")
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)