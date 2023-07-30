import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request

app = Flask(__name__)

df = None  

def load_data():
    global df  
    df = pd.read_csv('D:/Capstone Project/Dataset/umkm_cleaned_v2.csv')

load_data() 

def recommended_umkm(user_category):
  datacat = df[df['category'].isin(user_category)]  
  datacat.reset_index(level=0, inplace=True) 

  # Converting the place name into vectors and used bigram
  tf = TfidfVectorizer()
  tfidf_matrix = tf.fit_transform(datacat['nama_usaha'])

  # Calculating the similarity measures based on Cosine Similarity
  sg = cosine_similarity(tfidf_matrix, tfidf_matrix)

  # Get the index corresponding to produk
  sig = list(enumerate(sg[datacat.index]))
  sig = sorted(sig, key=lambda x: x[1][0], reverse=True)
  sig = sig[0:20]
  produk_indices = [i[0] for i in sig]

  # Top 20 recommendation
  rec = datacat.iloc[produk_indices, 1:]
  return rec.to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    category = request.args.get('category')
    user_category = category.split(',')

    recommendation_umkm = recommended_umkm(user_category)

    return jsonify(recommendation_umkm)

if __name__ == '__main__':
    app.run(debug=True)