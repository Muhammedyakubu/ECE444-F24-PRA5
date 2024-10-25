from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

loaded_model = None
vectorizer = None

def load_model():
    global loaded_model, vectorizer
    with open("basic_classifier.pkl", "rb") as fid:
        loaded_model = pickle.load(fid)
    with open("count_vectorizer.pkl", "rb") as fid:
        vectorizer = pickle.load(fid)

load_model()

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

def make_prediction(text):
    return loaded_model.predict(vectorizer.transform([text]))[0]

@application.route("/predict", methods=["GET"])
def predict():
  prediction = make_prediction(request.args.get("text"))
  print(request.args.get("text"), prediction)
  return jsonify({"prediction": prediction})



if __name__ == "__main__":
    load_model()
    application.run(port=5000, debug=True)
