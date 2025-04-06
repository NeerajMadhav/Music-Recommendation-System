from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import urllib.parse

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('spotify_data.csv')

# Use the 'year' column directly, assuming it already contains the release year
data['release_year'] = data['year']

# Select the relevant features for recommendation
features = ['popularity', 'loudness', 'instrumentalness', 'tempo', 'valence', 'energy', 'danceability']

# Handle the genre with One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
genre_encoded = encoder.fit_transform(data[['genre']])

# Combine the encoded genre with the numerical features
X_full = pd.concat([pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out()), data[features]], axis=1)

# Ensure all column names are strings
X_full.columns = X_full.columns.astype(str)

# Normalize (scale) the full feature set
scaler_full = MinMaxScaler()
X_scaled_full = scaler_full.fit_transform(X_full)

# Normalize (scale) only the numerical features
scaler_numeric = MinMaxScaler()
X_numeric = data[features]
X_scaled_numeric = scaler_numeric.fit_transform(X_numeric)

# Build and train the KNN models
knn_model_full = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_model_full.fit(X_scaled_full)

knn_model_numeric = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_model_numeric.fit(X_scaled_numeric)

# Function to filter data based on song release preference
def filter_by_song_type(song_type_choice, data):
    current_year = datetime.now().year
    if song_type_choice == "latest":
        filtered_data = data[data['release_year'] >= current_year - 5]
    elif song_type_choice == "old":
        filtered_data = data[data['release_year'] < current_year - 5]
    else:
        filtered_data = data
    return filtered_data

# Function to recommend songs based on user input
def recommend_song(user_input, knn_model, scaler):
    # Transform user input to the same scale as the training data
    user_input_scaled = scaler.transform([user_input])

    # Find the nearest songs using the KNN model
    distances, indices = knn_model.kneighbors(user_input_scaled)

    # Return the recommended songs
    recommendations = data.iloc[indices[0]]
    return recommendations[['artist_name', 'track_name', 'popularity', 'genre', 'release_year']]

# Function to generate YouTube search URL
def generate_youtube_search_url(query):
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    return search_url

# Function to generate Spotify search URL
def generate_spotify_search_url(query):
    search_url = f"https://open.spotify.com/search/{urllib.parse.quote(query)}"
    return search_url

@app.route('/')
def index():
    genres = data['genre'].unique()
    return render_template('index.html', genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input values from the form
    genre_choice = request.form['genre']
    popularity = float(request.form['popularity'])
    loudness = float(request.form['loudness'])
    instrumentalness = float(request.form['instrumentalness'])
    tempo = float(request.form['tempo'])
    valence = float(request.form['valence'])
    energy = float(request.form['energy'])
    danceability = float(request.form['danceability'])
    song_type = request.form['song_type']

    # Filter data based on song type
    filtered_data = filter_by_song_type(song_type, data)

    if filtered_data.empty:
        return render_template('recommendations.html', recommendations="<p>No songs found for the selected criteria.</p>", videos=[])

    # Handle genre encoding
    if genre_choice != "all":
        # Filter the dataset based on selected genre
        filtered_data = filtered_data[filtered_data['genre'] == genre_choice]
        if filtered_data.empty:
            return render_template('recommendations.html', recommendations="<p>No songs found for the selected genre and criteria.</p>", videos=[])

        # Re-encode genres for the filtered data
        genre_encoded_filtered = encoder.transform(filtered_data[['genre']])
        X_filtered_full = pd.concat([pd.DataFrame(genre_encoded_filtered, columns=encoder.get_feature_names_out()), filtered_data[features]], axis=1)

        # Scale the filtered data
        X_scaled_filtered_full = scaler_full.transform(X_filtered_full)

        # Update the KNN model with the filtered data
        knn_model_filtered = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn_model_filtered.fit(X_scaled_filtered_full)

        # Prepare user input
        genre_encoded_input = encoder.transform([[genre_choice]])
        user_input = list(genre_encoded_input[0]) + [popularity, loudness, instrumentalness, tempo, valence, energy, danceability]

        # Recommend songs
        recommendations = recommend_song(user_input, knn_model_filtered, scaler_full)
    else:
        # Use only numerical features
        user_input = [popularity, loudness, instrumentalness, tempo, valence, energy, danceability]

        # Check if there are any numerical features to scale
        if not features:
            return render_template('recommendations.html', recommendations="<p>No numerical features available.</p>", videos=[])

        # Scale the user input
        user_input_scaled = scaler_numeric.transform([user_input])

        # Recommend songs
        recommendations = recommend_song(user_input, knn_model_numeric, scaler_numeric)

    # Check if recommendations are empty
    if recommendations.empty:
        return render_template('recommendations.html', recommendations="<p>No songs found matching your preferences.</p>", videos=[])

    # Add YouTube and Spotify search URLs to recommendations DataFrame
    recommendations['youtube_link'] = recommendations.apply(
        lambda row: generate_youtube_search_url(f"{row['artist_name']} {row['track_name']}"), axis=1
    )
    recommendations['spotify_link'] = recommendations.apply(
        lambda row: generate_spotify_search_url(f"{row['artist_name']} {row['track_name']}"), axis=1
    )

    # Convert recommendations to HTML table
    recommendations_html = recommendations.to_html(index=False, escape=False, formatters={
        'youtube_link': lambda x: f'<a href="{x}" target="_blank">YouTube</a>',
        'spotify_link': lambda x: f'<a href="{x}" target="_blank">Spotify</a>'
    })

    return render_template('recommendations.html', recommendations=recommendations_html)

if __name__ == '__main__':
    app.run(debug=True)
