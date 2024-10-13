from flask import Flask, jsonify, request
import joblib
import traceback
import logging
import re
from nltk.corpus import stopwords
import nltk

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

pipeline = joblib.load('./text_classification_pipeline.joblib')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove digits
    text = re.sub(r'\d', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    text = ' '.join(words)

    return text


@app.route('/')
def home():
    return "Welcome to your Zaniar API!"

@app.route('/api/', methods=['GET'])
def get_data():
    data = {"message": "This API only supports POST method! You can send your job description in Json format."}
    return jsonify(data)

@app.route('/api/', methods=['POST'])
def post_data():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')

        if not text:
            logging.warning('No text provided in the request')
            return jsonify({'error': 'No text provided'}), 400

        text = preprocess_text(text)

        prediction = pipeline.predict([text])
        prediction = prediction[0]
        print(prediction)
        print(type(prediction))
        if prediction == 0:
            prediction = "Seems legit!!"
        else:
            prediction = "be careful with this one!!"

        logging.info(f'Prediction: {prediction}')

        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        logging.error('Error during prediction')
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    print('Run2')
    app.run(host='0.0.0.0', port=5000)
