from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle

app = Flask(__name__)

# Load the CSV file
movies = pd.read_csv('youtube.csv', encoding='ISO-8859-1')

# Combine title, category, and comments into a single text column
movies['Combined_Text'] = movies['title'].fillna('') + ' ' + movies['Category'].fillna('') + ' ' + movies['Hahstag'].fillna('')

# Apply TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['Combined_Text']).toarray()

# Convert dense matrix to sparse matrix
sparse_tfidf_matrix = csr_matrix(tfidf_matrix)

# Calculate cosine similarity directly
similarity_matrix = cosine_similarity(sparse_tfidf_matrix)

# Define recommend function based on combined text
def recommend(input_text, num_recommendations=10):
    input_vector = tfidf_vectorizer.transform([input_text]).toarray()
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    similar_indices = similarity_scores.argsort()[0][::-1]  # Highest similarity first
    
    recommended_movies = []
    for idx in similar_indices:
        if idx != 0:  # Exclude the input movie itself
            recommended_movies.append(movies.iloc[idx]['Category'])
            if len(recommended_movies) >= num_recommendations:
                break
    
    return recommended_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_categories = []
    if request.method == 'POST':
        input_text = request.form['input_text']
        recommended_categories = recommend(input_text, num_recommendations=10)
    return render_template('index.html', recommended_categories=recommended_categories)

if __name__ == '__main__':
    app.run(debug=True)
