from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

PDF_PATH = "uploads/resume.pdf"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# Read PDF
reader = PdfReader(PDF_PATH)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# Chunk text
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Vectorize
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(chunks)

# Save
with open("data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

with open("data/vectors.pkl", "wb") as f:
    pickle.dump((vectorizer, vectors), f)

print("✅ Resume indexed successfully")