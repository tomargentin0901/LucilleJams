import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from typing import Dict
import pandas as pd
from models.embeddings import Word2VecSongsEmbeddings, Word2VecTitlesEmbeddings
import json
from tqdm import tqdm
from utils.dataset_utils import clean_text
import faiss
from utils.inference_utils import next_K_songs


def get_songs_submission(
    title_embeddings_model: Word2VecTitlesEmbeddings,
    songs_embeddings_model: Word2VecSongsEmbeddings,
    faiss: faiss.swigfaiss.Index,
    options: Dict[str, int],
    challenge_set_path: str = "spotify_million_playlist_dataset_challenge/challenge_set.json",
    save: bool = True,
    K: int = 500,
    n_playlists_to_pull: int = 300,
    strategy: str = "most_popular_songs",
    weighted: bool = True,
    pool_method: str = "median",
):
    """
    Generates a submission of recommended songs for each playlist in the challenge set.

    Args:
        df (pd.DataFrame): The DataFrame containing the song data.
        title_embeddings_model (Word2VecTitlesEmbeddings): The Word2Vec model for title embeddings.
        songs_embeddings_model (Word2VecSongsEmbeddings): The Word2Vec model for song embeddings.
        faiss (faiss.swigfaiss.Index): The Faiss index for efficient similarity search.
        options (Dict[str, int]): The dictionary of options for choosing song granularity.
        challenge_set_path (str, optional): The path to the challenge set JSON file. Defaults to "spotify_million_playlist_dataset_challenge/challenge_set.json".
        save (bool, optional): Whether to save the submission as a CSV file. Defaults to True.
        K (int, optional): The number of songs to recommend for each playlist. Defaults to 500.
        strategy (str, optional): The strategy for selecting songs. Defaults to "most_popular_songs".

    Returns:
        pd.DataFrame: The submission DataFrame containing the recommended songs for each playlist.
    """
    with open(challenge_set_path) as f:
        challenge_set = json.load(f)
    submission = []
    for playlist in tqdm(challenge_set["playlists"]):
        pid = playlist["pid"]
        if "name" in playlist:
            title = clean_text(playlist["name"])
        else:
            title = ""
        tracks = playlist["tracks"]  # Playlist to complete
        songs = []
        seeds = []
        for song in tracks:  # Get songs according the chosen granularity
            song = [song["artist_name"], song["track_name"], song["album_name"]]
            seeds.append(" - ".join(song))
            song = {
                key: song[i].strip()
                for i, (key, val) in enumerate(options.items())
                if val == 1
            }
            song = " - ".join(song.values())
            songs.append(song)
        n_most_sim_artists_to_consider = 50
        n_playlists_to_pull = n_playlists_to_pull
        playlist_completed = []
        while len(playlist_completed) != K:
            playlist_completed, _ = next_K_songs(
                seeds,
                title,
                songs,
                faiss,
                title_embeddings_model,
                songs_embeddings_model,
                K=K,
                n_playlists_to_pull=n_playlists_to_pull,
                strategy=strategy,
                weighted=weighted,
                pool_method=pool_method,
                n_most_sim_artists_to_consider=n_most_sim_artists_to_consider,
            )
            n_most_sim_artists_to_consider += 50
            n_playlists_to_pull += 100
        n_most_sim_artists_to_consider = 50
        n_playlists_to_pull = 300
        if len(playlist_completed) != K:
            raise ValueError("Playlist not completed.")
        playlist_completed.insert(0, int(pid))
        submission.append(playlist_completed)
    cols = ["pid"] + [f"track_uri_{i}" for i in range(1, K + 1)]
    submission = pd.DataFrame(submission, columns=cols)
    if save:
        submission.to_csv(
            "submission_songs.csv", index=False
        )  # Save the submission as songs and not as tracks, helps to see the relevance of the recommendations
    return submission


def get_final_submission(
    songs_to_id: Dict[str, str],
    path_to_songs_submission: str = "submission_songs.csv",
    save: bool = True,
    K: int = 500,
):

    submission = pd.read_csv(path_to_songs_submission)
    for col in [f"track_uri_{i}" for i in range(1, K + 1)]:
        submission[col] = submission[col].map(songs_to_id)
    if save:
        submission.to_csv("submission.csv", index=False)
    return submission
