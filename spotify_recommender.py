# Importing libraries
import os
from dotenv import load_dotenv 
load_dotenv('pass.env')

# Spotipy library
import spotipy 
import pandas as pd 
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np 

# Credentials
spotify_client_id = os.getenv ('spotify_client_id')
spotify_client_secret = os.getenv ('spotify_client_secret')

# Initialize SpotiPy
sp = spotipy.Spotify(auth_manager = SpotifyClientCredentials (spotify_client_id, spotify_client_secret))

def audio():
    artist = input("Select an artist to recommend one of his songs based on the features of your previous song: ")
    results = sp.search(q=f'artist: {artist}', limit=50)

    track_ids = [track['id'] for track in results['tracks']['items']]
    song_names = [track['name'] for track in results['tracks']['items']]
 
    audio_features = sp.audio_features(track_ids)

    df = pd.DataFrame(audio_features)
    df['artist'] = artist
    df['song_name'] = song_names
    return df

# Recommender
def recommender():
    song_name = input('Choose a song: ')
    df = audio()
 
    x = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

    # standarize the data
    scaler = StandardScaler()
    x_prep = scaler.fit_transform(x)
    kmeans = KMeans(n_clusters=9, random_state=42)
    kmeans.fit(x_prep)
    clusters = kmeans.predict(x_prep)

    #create new dataframe with title, artist and cluster assigned
    scaled_df = pd.DataFrame(x_prep, columns=x.columns)
    scaled_df['song_name'] = df["song_name"]
    scaled_df['artist'] = df['artist']
    scaled_df['cluster'] = clusters

    #SONG RECOMMENDATON
    
    # get song id
    
    results = sp.search(q=f'track:{song_name}', limit=1)
    track_id = results['tracks']['items'][0]['id']

    right = input(f'Is {results["tracks"]["items"][0]["name"]} by {results["tracks"]["items"][0]["artists"][0]["name"]} your song?')
    if right.lower() not in ["yes", "y", "si", "affirmative"]:
        print("We could not find your song, try another.")
        return
     
    # get song features with the obtained id
    audio_features = sp.audio_features(track_id)
    
    # create dataframe
    df_ = pd.DataFrame(audio_features)
    new_features = df_[x.columns]
    
    # scale features
    scaled_x = scaler.transform(new_features)
    
    # predict cluster
    cluster = kmeans.predict(scaled_x)
    
    # filter dataset to predicted cluster
    filtered_df = scaled_df[scaled_df['cluster'] == cluster[0]][x.columns]
    
    # get closest song from filtered dataset
    closest, _ = pairwise_distances_argmin_min(scaled_x, filtered_df)
    
    # return it in a readable way
    print('\n [RECOMMENDED SONG]')
    print(' - '.join([scaled_df.loc[closest]['song_name'].values[0], scaled_df.loc[closest]['artist'].values[0]]))
    return
