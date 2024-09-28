import sys
import os


from utils.inference_utils import next_K_songs, get_user_playlist_embedding
from utils.visualization_utils import get_neighbors_plot

from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import faiss
import dash_bootstrap_components as dbc

from typing import List, Union


import pandas as pd
from ast import literal_eval
import numpy as np
import pickle
from dotenv import load_dotenv
import requests
import base64
from pathlib import Path

import random

load_dotenv()


def get_model(model_path: Union[Path, str]):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# Playlist embeddings for neighbors plot #TODO: If too memory consuming store embeddings along the df and fetch it by querying
# embeddings = get_model("embeddings.pickle")
# Useful func

songs_to_id = get_model("songs_to_id.pkl")

# Get Spotify API Keys

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Get search space.

index = faiss.read_index("embeddings_faiss.index")


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
            "preview_url": track["preview_url"] if "preview_url" in track else "",
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


app = Dash(
    __name__, external_stylesheets=[dbc.themes.FLATLY], prevent_initial_callbacks=True
)
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
    id="submit-button-recommendation",
    n_clicks=0,
    style={
        "display": "block",
        "marginTop": "20px",
        "width": "15%",
        "marginRight": "10px",
    },
)

# Get neighbors embeddings

neighbors_playlists_button = dbc.Button(
    "Get Neighbors",
    id="submit-button-neighbors",
    n_clicks=0,
    style={"display": "block", "marginTop": "20px", "width": "15%"},
)

submit_buttons = dbc.Row(
    [recommendation_button, neighbors_playlists_button],
    style={"justify-content": "center"},
)

final_output = dbc.Row(
    dbc.Col(
        dbc.Spinner(
            id="loading_spinner",
            children=html.Div(id="final_output"),
            spinner_style={"margin-top": "50px"},
        ),
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
        submit_buttons,
        html.Br(),
        final_output,
        html.Br(),
    ]
)


@app.callback(
    Output("final_output", "children", allow_duplicate=True),
    Input("submit-button-neighbors", "n_clicks"),
    State("playlist-title", "value"),
    State("artist", "value"),
)
def update_neighbors(n_clicks, playlist_title, artists):
    if not playlist_title:
        playlist_title = ""
    if n_clicks > 0:
        user_playlist_embedding = get_user_playlist_embedding(
            playlist_title, artists, title_embeddings_model, songs_embeddings_model
        )
        fig = get_neighbors_plot(
            index,
            user_playlist_embedding,
            title_embeddings_model,
            playlist_title=f"{playlist_title if playlist_title else 'USER PLAYLIST'}",
            N=5000,
            perplexity=10,
            n_clusters=30,
        )
        return dcc.Graph(figure=fig)

    return None


# Callback for generating the recommended songs
@app.callback(
    Output("final_output", "children", allow_duplicate=True),
    Input("submit-button-recommendation", "n_clicks"),
    State("playlist-title", "value"),
    State("artist", "value"),
)
def update_table(n_clicks, playlist_title, artists, shuffle: bool = True):
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
        if shuffle:
            random.shuffle(playlist_completed)
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
                        html.Td(artist, style={"text-align": "center"}),
                        html.Td(song_title, style={"text-align": "center"}),
                        html.Td(album_name, style={"text-align": "center"}),
                        html.Td(preview, style={"text-align": "center"}),
                    ]
                )
            )

        table = dbc.Table(
            # Table Header
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Artist", style={"text-align": "center"}),
                            html.Th("Song Title", style={"text-align": "center"}),
                            html.Th("Album Name", style={"text-align": "center"}),
                            html.Th("Listen", style={"text-align": "center"}),
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


if __name__ == "__main__":
    app.run_server(debug=True)
