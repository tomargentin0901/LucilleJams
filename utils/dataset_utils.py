import os
import json
from typing import Dict, List, Iterable
from tqdm import tqdm
import re
import string
import emoji
import pandas as pd
from config.emoji_name_to_text import emoji_name_to_text
import pickle
from collections import Counter
from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from ast import literal_eval


emoji_pattern = "|".join(
    map(re.escape, emoji_name_to_text.keys())
)  # Create a regex pattern to match emojis


def tokenize_songs(
    df: pd.DataFrame,
    options: Dict[str, str],
    songs_col_name: str = "songs",
    remove_consequent_duplicates: bool = True,
) -> Iterable[List[str]]:
    """
    Tokenize the songs in the given DataFrame based on the specified options.
    Args:
        df (pd.DataFrame): The DataFrame containing the songs.
        options (Dict[str,str]): A dictionary specifying the options for tokenization.
        songs_col_name (str, optional): The name of the csolumn containing the songs. Defaults to "songs".

    Yields:
        Iterable[List[str]]: An iterable of tokenized songs.
    """
    for songs in df[songs_col_name]:
        tokenized_songs = []
        for song in songs:
            song = song.split(" - ")
            song = {
                key: song[i].strip()
                for i, (key, val) in enumerate(options.items())
                if val == 1
            }
            song = " - ".join(song.values())
            tokenized_songs.append(song)
        if remove_consequent_duplicates:
            tokenized_songs = [key for key, _ in groupby(tokenized_songs)]
        yield tokenized_songs


def songs_to_id(
    path: str = "spotify_million_playlist_dataset/data",
    save: bool = True,
    verbose: bool = True,
):
    """
    Returns a dictionary with the songs as keys and the ids as values
    """
    id_to_songs = spotify_songs_vocabulary(path)
    if verbose:
        seen = set()
        dup = [x for x in id_to_songs.values() if x in seen or seen.add(x)]
        print(f"{len(dup)} duplicates found \n Removing duplicates...")
    songs_to_id = {}
    seen = set()
    for key, value in id_to_songs.items():
        if value not in seen:
            songs_to_id[value] = key
            seen.add(value)
    if save:
        with open("songs_to_id.pkl", "wb") as f:
            pickle.dump(songs_to_id, f)
    return songs_to_id


def get_songs_from_json(json_path: str) -> Dict[str, str]:
    """keep track of the songs in the json file, mapping the track_uri to the track_name
    Args:
        json_path (str): path to the spotify json slice to extract songs from
    Returns:
        Dict[str,str]: dictionary mapping track_uri to track_name
    """
    file = open(json_path)
    data = json.load(file)
    songs = {}
    for playlist in data["playlists"]:
        for track in playlist["tracks"]:
            songs[track["track_uri"]] = (
                track["artist_name"]
                + " - "
                + track["track_name"]
                + " - "
                + track["album_name"]
            )
    file.close()
    return songs


def spotify_songs_vocabulary(file_path: str) -> Dict[str, str]:
    """track url to songs name for every songs of the spotify dataset
    Args:
        file_path (str): path to the spotify dataset
    Returns:
        Dict[str, str]: dictionary mapping track_uri to track_name for every songs of the spotify dataset
    """

    track_id_to_song = {}
    if os.path.isfile(file_path):
        track_id_to_song = get_songs_from_json(file_path)
    else:
        for file in tqdm(os.listdir(file_path)):
            if file.endswith(".json"):
                track_id_to_song.update(
                    get_songs_from_json(os.path.join(file_path, file))
                )
    return track_id_to_song


def get_playlist_titles_from_json(json_path: str) -> List[str]:
    """Get all the playlist titles from a spotify dataset slice json file
    Args:
        json_path (str): path to the spotify dataset slice json file
    Returns:
        List[str]: list of playlist titles
    """
    file = open(json_path)
    data = json.load(file)
    playlist_titles = []
    for playlist in data["playlists"]:
        playlist_titles.append(playlist["name"])
    file.close()
    return playlist_titles


def spotify_playlist_titles_vocabulary(file_path: str) -> List[str]:
    """Get all the playlist titles from the spotify dataset
    Args:
        file_path (str): path to the spotify dataset
    Returns:
        List[str]: list of all the playlist titles
    """

    playlist_titles = []
    if os.path.isfile(file_path):
        playlist_titles = get_playlist_titles_from_json(file_path)
    else:
        for file in tqdm(os.listdir(file_path)):
            if file.endswith(".json"):
                playlist_titles.extend(
                    get_playlist_titles_from_json(os.path.join(file_path, file))
                )
    return list(set(playlist_titles))


def clean_emojis(text: str) -> str:
    """Remove emojis from text. Some playlists titles contain emojis, we need to remove them and replace them with the corresponding text.
    Args:
        text (str): text to replace emojis from
    Returns:
        text (str): text with emojis replaced by their corresponding text
    """

    # Check if the text contains emojis
    if bool(emoji.emoji_count(text)):
        text = " ".join(emoji.demojize(text).split(":")).strip()
    return text


def convey_emoji(text, emoji_pattern):
    return re.sub(emoji_pattern, lambda match: emoji_name_to_text[match.group(0)], text)


def remove_special_characters(text):
    """Remove special characters from text."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def to_lower_case(text):
    """Convert text to lower case."""
    return text.lower()


def remove_whitespace(text):
    """Remove extra whitespaces from text."""
    return " ".join(text.split())


def remove_punctuation(text):
    """Remove punctuation from text."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def clean_text(text):
    """Apply all text cleaning operations to text."""
    text = clean_emojis(text)
    text = remove_special_characters(text)
    text = to_lower_case(text)
    text = remove_whitespace(text)
    text = remove_punctuation(text)
    text = convey_emoji(text, emoji_pattern)
    return text


def create_df_from_spotify_playlists(
    json_files_path: str, columns_names: List[str] = None
) -> pd.DataFrame:
    """
    Create a DataFrame from the spotify dataset.
    It will contain the playlist title, the songs sequence and the number of followers for each playlist.
    It'll be used for creating the training dataset.
    Args:
        json_files_path(str): path to the spotify dataset
        columns_names (List[str], optional): columns names of the df to build. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame containing the playlist title, the songs sequence and the number of followers for each playlist.
    """

    data = []
    if columns_names is None:
        columns_names = ["title", "songs", "popularity"]
    for mpd_slide in os.listdir(json_files_path):
        file = open(os.path.join(json_files_path, mpd_slide))
        json_data = json.load(file)
        for playlist in json_data["playlists"]:
            playlist_title = playlist["name"]
            playlist_title = clean_text(playlist_title)
            tracks = [
                song["artist_name"]
                + " - "
                + song["track_name"]
                + " - "
                + song["album_name"]
                for song in playlist["tracks"]
            ]
            num_followers = playlist["num_followers"]
            data.append([playlist_title, tracks, num_followers])
        file.close()
    df = pd.DataFrame(data, columns=columns_names)
    return df


def create_and_save_df_from_spotify_playlists(
    json_files_path: str, save_path: str, columns_names: List[str] = None
) -> pd.DataFrame:
    """
    Create a DataFrame from the spotify dataset and save it.
    It will contain the playlist title, the songs sequence and the number of followers for each playlist.
    It'll be used for creating the training dataset.
    Args:
        json_files_path(str): path to the spotify dataset
        save_path(str): path to save the df
        columns_names (List[str], optional): columns names of the df to build. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame containing the playlist title, the songs sequence and the number of followers for each playlist.
    """

    df = create_df_from_spotify_playlists(json_files_path, columns_names)
    df.to_csv(save_path, index=False)
    return df


def get_artists_frequency_stats(df: pd.DataFrame):
    """
    Calculate the frequency statistics of artists in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the songs and artists.

    Returns:
    pd.DataFrame: A DataFrame containing the quantiles of artist frequencies.

    """
    df["artists"] = df["songs"].apply(lambda x: [song.split(" - ")[0] for song in x])
    all_artists = [artist for sublist in df["artists"].tolist() for artist in sublist]
    artists_count = Counter(all_artists)
    # Plot the frequency distribution
    artist_frequencies = list(artists_count.values())
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(artist_frequencies, bins=range(1, 50))
    df = pd.DataFrame.from_dict(artists_count, orient="index", columns=["frequency"])
    quantiles = df.quantile([i / 10 for i in range(1, 10)])
    return quantiles


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize the embeddings."""
    embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.where(
        embeddings_norm != 0, embeddings / embeddings_norm, embeddings
    )
    return embeddings


def query_data_by_index(
    indices, col_songs_name: str = "songs", dbname: str = "spotify_dataset.db"
):
    """Fetch most similar playlists from a list of indices"""
    conn = sqlite3.connect(dbname)
    placeholders = ", ".join(["?"] * len(indices))
    query = f'SELECT * FROM spotify_data WHERE "index" IN ({placeholders})'
    df = pd.read_sql(query, conn, params=indices)
    df[col_songs_name] = df[col_songs_name].apply(literal_eval)
    conn.close()
    return df
