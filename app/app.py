import sys
import os


from models.embeddings import (
    Word2VecSongsEmbeddings,
    Word2VecTitlesEmbeddings,
)
from utils.inference_utils import next_K_songs

from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import faiss
import dash_bootstrap_components as dbc

import json
from typing import List, Union


import pandas as pd
from ast import literal_eval
import numpy as np
import pickle
from dotenv import load_dotenv
import requests
import base64
from pathlib import Path


load_dotenv()


def get_model(model_path: Union[Path, str]):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# Useful func

songs_to_id = get_model("songs_to_id.pkl")

# Get Spotify API Keys

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")


# Function to get the access token
def get_access_token(client_id, client_secret):

    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    data = {"grant_type": "client_credentials"}

    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    token_url = "https://accounts.spotify.com/api/token"
    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        token_info = response.json()
        return token_info.get("access_token")
    else:
        print(f"Error: Received status code {response.status_code}")
        print(f"Error message: {response.text}")
        return None


def get_track_id(track_id: str):
    return track_id.split("track:")[-1]


def get_url_overview_from_recommended_songs(recommended_songs: List[str]) -> List[str]:
    url_overviews = []
    for song in recommended_songs:
        track_id = get_track_id(songs_to_id[song])
        url = get_track_metadata(track_id, access_token)["preview_url"]
        url_overviews.append(url)
    return url_overviews


# Function to split the data ensuring exactly 3 parts
def split_song(entry):
    parts = entry.split(" - ")
    if len(parts) > 3:
        # Join the last two parts
        parts = [parts[0], parts[1], " - ".join(parts[2:])]
    return parts


def get_track_metadata(track_id: str, access_token):

    # Define the URL and headers for the API request
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    # Make the GET request to the Spotify API
    response = requests.get(url, headers=headers)

    # Check the response status code and handle the response
    if response.status_code == 200:
        # Successfully retrieved data
        track = response.json()
        metadata = {
            "name": track["name"],
            "album": track["album"]["name"],
            "artists": [artist["name"] for artist in track["artists"]],
            "release_date": track["album"]["release_date"],
            "duration_ms": track["duration_ms"],
            "preview_url": track["preview_url"],
        }
        return metadata
    else:
        # Handle error
        print(f"Error: Received status code {response.status_code}")
        print(f"Error message: {response.text}")
        return {}


# Spotify API token

access_token = get_access_token(client_id, client_secret)

# Get models

songs_embeddings_model = get_model(
    Path(__file__).parent.parent / Path("models/songs_model.pickle")
)
title_embeddings_model = get_model(
    Path(__file__).parent.parent / Path("models/title_model.pickle")
)

# Get search space.

index = faiss.read_index("embeddings_faiss.index")


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "LucilleJams"


logo = html.Img(
    src=app.get_asset_url("logo.png"), style={"width": "20%", "height": "auto"}
)

header = dbc.Row(
    logo,
    style={"justify-content": "center", "text-align": "center"},
)

line = html.Hr(
    style={
        "border": "none",
        "border-top": "3px solid #bbb",  # Light gray border
        "width": "80%",  # Width of the line
        "margin": "20px auto",  # Center horizontally and add top/bottom margin
        "opacity": "0.6",  # Slight transparency for a softer look
    }
)

# Title playlist selection

title_label = dbc.Col(
    html.Label("Playlist Title"), width=3, style={"text-align": "center"}
)
title_input = dbc.Col(
    dbc.Input(
        id="playlist-title", placeholder="Choose a playlist title...", type="text"
    ),
    width=6,
)

playlist_title = dbc.Row([title_label, title_input], style={"justify": "center"})

# Artists selection

select_artists_label = dbc.Col(
    html.Label("Select artists"), width=3, style={"text-align": "center"}
)
select_artists_dropdown = dbc.Col(
    dcc.Dropdown(
        id="artist",
        options=songs_embeddings_model.model.wv.index_to_key,
        value=[],
        multi=True,
        placeholder="Select artists you like...",
    ),
    width=6,
)

select_artists = dbc.Row(
    [select_artists_label, select_artists_dropdown],
    style={"justify": "center", "margin-top": "20px"},
)

# Get recommendation

recommendation_button = dbc.Button(
    "Get Recommendations",
    id="submit-button",
    n_clicks=0,
    style={"display": "block", "margin": "0 auto", "marginTop": "20px", "width": "20%"},
)


recommended_musics_board = dbc.Row(
    dbc.Col(
        html.Div(id="recommended_music_boards"),
        width=10,
        style={
            "height": "400px",  # Set a fixed height for the table
            "overflowY": "auto",  # Enable vertical scrolling
        },
    ),
    justify="center",
)

app.layout = html.Div(
    [
        header,
        line,
        html.Br(),
        playlist_title,
        html.Br(),
        select_artists,
        html.Br(),
        recommendation_button,
        html.Br(),
        recommended_musics_board,
    ]
)


# Callback for generating the recommended songs
@app.callback(
    Output("recommended_music_boards", "children"),
    Input("submit-button", "n_clicks"),
    State("playlist-title", "value"),
    State("artist", "value"),
)
def update_table(n_clicks, playlist_title, artists):
    if not playlist_title:
        playlist_title = ""
    if n_clicks > 0:
        playlist_completed, _ = next_K_songs(
            [],
            playlist_title,
            artists,
            index,
            title_embeddings_model,
            songs_embeddings_model,
            K=50,
        )
        url_overviews = get_url_overview_from_recommended_songs(playlist_completed)
        print(url_overviews)
        data = []
        for song, url in zip(playlist_completed, url_overviews):
            artist, song_title, album_name = split_song(song)
            if not url:
                preview = "No Preview Available"
            else:
                preview = html.Audio(src=url, controls=True)

            data.append(
                html.Tr(
                    [
                        html.Td(artist),
                        html.Td(song_title),
                        html.Td(album_name),
                        html.Td(preview),
                    ]
                )
            )

        table = dbc.Table(
            # Table Header
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Artist"),
                            html.Th("Song Title"),
                            html.Th("Album Name"),
                            html.Th("Listen"),
                        ]
                    )
                )
            ]
            +
            # Table Body
            [html.Tbody(data)],
            bordered=True,
            striped=True,
            hover=True,
            responsive=True,
        )

        return table

    return None


# @app.callback(
#     Output("recommended_music_boards", "data"),
#     Input("submit-button", "n_clicks"),
#     State("playlist-title", "value"),
#     State("artist", "value"),
# )
# def print_data(n_clicks, playlist_title, artists):
#     if not playlist_title:
#         playlist_title = ""
#     if n_clicks > 0:
#         print(playlist_title, artists)
#         playlist_completed, _ = next_K_songs(
#             [],
#             playlist_title,
#             artists,
#             index,
#             title_embeddings_model,
#             songs_embeddings_model,
#             K=50,
#         )
#         url_overviews = get_url_overview_from_recommended_songs(playlist_completed)
#         print(url_overviews)
#         data = []
#         for song, url in zip(playlist_completed, url_overviews):
#             artist, song_title, album_name = split_song(song)
#             if not url:
#                 url = "No Preview Available"
#             else:
#                 url = f'<audio controls src="{url}"></audio>'

#             data.append(
#                 {
#                     "Artist": artist,
#                     "Song Title": song_title,
#                     "Album Name": album_name,
#                     "Preview url": url,
#                 }
#             )
#         recommended_songs_df = pd.DataFrame(data)
#         print(recommended_songs_df)
#         return recommended_songs_df.to_dict("records")


if __name__ == "__main__":
    app.run_server(debug=True)
