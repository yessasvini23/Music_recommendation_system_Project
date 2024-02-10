#!/usr/bin/env python
# coding: utf-8

# Importing the necessary packages.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import PercentFormatter
from scipy.stats import zscore


# Reading the file

# In[ ]:


df=pd.read_csv(r"C:\Users\suhas\OneDrive\Documents\universal_top_spotify_songs.csv",encoding="unicode_escape")





# In[ ]:


from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from surprise import dump
import joblib



# Create a Surprise dataset
reader = Reader(rating_scale=(0, 1000))
data = Dataset.load_from_df(df[['name', 'artists', 'popularity']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build a collaborative filtering model (using SVD as an example)
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f'Model RMSE: {rmse}')

# Save the model
model_filename = 'collaborative_model.joblib'
dump.dump(model_filename, algo=model)


# In[ ]:


import streamlit as st
from surprise import dump, Dataset, Reader
import pandas as pd

# Load the trained model
model_filename = 'collaborative_model.joblib'
model = dump.load(model_filename)[1]

# Sample dataset (replace this with your actual dataset)
df = pd.DataFrame({
    'song_id': ['Cruel Summer', 'Lovin On Me', 'My Love Mine All Mine', 'Si No Estás', 'LALA',
                'Rockin\' Around The Christmas Tree', 'Seven (feat. Latto) (Explicit Ver.)',
                'Standing Next to You', 'Strangers'],
    'user_id': ['user1'] * 9,  # Replace with your actual user IDs
    'danceability': [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.7, 0.9]  # Replace with your actual danceability values
})

# Streamlit app header
st.title('Music Recommendation App')

# Input form to get user preferences
danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.slider('Energy', 0.0, 1.0, 0.5)
loudness = st.slider('Loudness', -20.0, 0.0, -10.0)

# Button to trigger recommendations
if st.button('Get Recommendations'):
    # User preferences
    user_input = {'user_id': 'user1', 'danceability': danceability, 'energy': energy, 'loudness': loudness}

    # Convert the user preferences to a Surprise Dataset
    reader = Reader(rating_scale=(0, 1))  # Update rating scale based on your dataset
    data = Dataset.load_from_df(df[['user_id', 'song_id', 'danceability']], reader)

    # Build the training set
    trainset = data.build_full_trainset()

    # Train the model
    model.fit(trainset)

    # Use the test method to generate predictions for all songs
    predictions = [model.predict(user_input['user_id'], song_id, user_input['danceability']) for song_id in df['song_id']]

    # Get the recommended song with the highest predicted rating
    top_recommendation = max(predictions, key=lambda x: x.est)

    # Display the recommended song
    st.success(f"Recommended Song: {top_recommendation.iid} with Predicted Rating: {top_recommendation.est}")

