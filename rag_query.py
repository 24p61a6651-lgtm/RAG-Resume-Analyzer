import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("data/vectors.pkl", "rb") as f:
    vectorizer, vectors = pickle.load(f)

while True:
    query = input("\nAsk a question (or type exit): ")
    if query.lower() == "exit":
        break

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors)[0]

    best_match = chunks[scores.argmax()]
    print("\nAnswer:\n", best_match)
