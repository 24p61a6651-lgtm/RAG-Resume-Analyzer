from flask import Flask, render_template, request
import os
import subprocess
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["resume"]
    path = os.path.join(UPLOAD_FOLDER, "resume.pdf")
    file.save(path)

    subprocess.run([sys.executable, "ingest.py"], check=True)

    return render_template("index.html", answer="Resume uploaded and indexed successfully!")

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form["question"]

    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("data/vectors.pkl", "rb") as f:
        vectorizer, vectors = pickle.load(f)

    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, vectors)[0]

    answer = chunks[scores.argmax()]

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)