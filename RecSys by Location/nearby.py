import pandas as pd 
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

df = None  

def load_data():
    global df  
    df = pd.read_csv('D:/Capstone Project/Dataset/umkm_cleaned_v2.csv')

load_data() 

def nearby_umkm(df, latitude, longitude):
    user_location = np.array([float(latitude), float(longitude)])

    df['distance'] = np.linalg.norm(df[['latitude', 'longitude']].values - user_location, axis=1)
    df = df.sort_values(by='distance')

    nearby_umkm = df[['nama_usaha', 'Deskripsi_usaha', 'category', 'address', 'no_hp', 'latitude', 'longitude']].head(20)
    return nearby_umkm.to_dict(orient='records')

@app.route('/nearby', methods=['POST'])
def nearby():
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    recommendation_nearby_umkm = nearby_umkm(df, latitude, longitude)

    return jsonify(recommendation_nearby_umkm)

if __name__ == '__main__':
    app.run(debug=True)