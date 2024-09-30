import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd
from models.embeddings import Word2VecSongsEmbeddings, Word2VecTitlesEmbeddings
import json
from tqdm import tqdm
from utils.dataset_utils import clean_text
import faiss
from utils.dataset_utils import normalize_embeddings
from typing import Callable
from jaro import jaro_winkler_metric
from utils.dataset_utils import query_data_by_index
import math


def get_playlist_recommendations(
    user_playlist_embedding: np.ndarray,
    playlist_title: str,
    faiss_index,
    n_playlists_to_pull: int,
    str_similarity_metric: Callable = jaro_winkler_metric,
):
    """
    Retrieves playlist recommendations based on user preferences.

    Computes the cosine similarity between the user playlist embedding and all the embeddings in the index.
    If the user playlist embedding is not NaN, it means that the title and songs of the playlist are known by the models.
    In this case, it retrieves the most similar playlists based on the cosine similarity scores.

    If the user playlist embedding is NaN, it means that both the title and songs of the playlist are not known by the models.
    In this case, it computes the most similar playlist by only looking at the title and computing fuzzy similarity.

    Args:
        user_playlist_embedding (np.ndarray): The embedding vector representing the user's playlist.
        playlist_title (str): The title of the user's playlist.
        faiss_index (faiss.swigfaiss.Index): The Faiss index used for similarity search.
        n_playlists_to_pull (int): The number of playlists to retrieve as recommendations.
        str_similarity_metric (Callable, optional): The string similarity metric used for title matching.
            Defaults to jaro_winkler_metric.

    Returns:
        tuple: A tuple containing the similarity scores and the DataFrame of recommended playlists.
    """

    if user_playlist_embedding.any():
        sim, idx = faiss_index.search(
            user_playlist_embedding.astype(np.float32), n_playlists_to_pull
        )
        sim, idx = sim[0], idx[0]
    else:
        sim, idx = lookup_df_by_title_similarity(
            playlist_title, n_playlists_to_pull, str_similarity_metric
        )
    idx = [int(id_) for id_ in idx]
    lookup_df = query_data_by_index(list(idx))
    return sim, lookup_df


def get_user_playlist_embedding(
    playlist_title: str,
    songs: List[str],
    title_embeddings_model: Word2VecTitlesEmbeddings,
    songs_embeddings_model: Word2VecSongsEmbeddings,
    pool_method: str = "median",
    alpha: float = 0.7,  # 30% to title and 70% to the songs.
):
    """
    Get the user playlist embedding by a weighted concatenation of the title and artist embeddings.
    We don't want the title to have a bigger impact than the music the playlist is made of, so we give it a lower weight.
    Args:
        playlist_title (str): The title of the playlist.
        songs (List[str]): List of songs in the playlist.
        title_embeddings_model (Word2VecTitlesEmbeddings): The Word2Vec model for title embeddings.
        songs_embeddings_model (Word2VecSongsEmbeddings): The Word2Vec model for song embeddings.
        pool_method (str, optional): The method used to pool the artist embeddings. Defaults to "median".
        title_weight (float, optional): The weight assigned to the title embeddings. Defaults to 0.2.
        songs_weight (float, optional): The weight assigned to the artist embeddings. Defaults to 0.8.

    Returns:
        numpy.ndarray: The user playlist embedding.
    """
    title_emb = title_embeddings_model.get_titles_embeddings(
        playlist_title, title_embeddings_model.model
    )
    title_emb = normalize_embeddings(title_emb)
    title_emb *= alpha / (math.sqrt(alpha**2 + 1))

    artist_emb = songs_embeddings_model.get_songs_playlists_embeddings(
        songs, songs_embeddings_model.model, pool_method
    )  # Get the artist embeddings
    artist_emb = normalize_embeddings(artist_emb)
    artist_emb *= 1 / math.sqrt(alpha**2 + 1)

    user_playlist_embedding = np.concatenate(
        (title_emb, artist_emb), axis=1
    )  # Concatenate the title and artist embeddings, we don't want the title to have a big impact on the final embedding

    return user_playlist_embedding


def lookup_df_by_title_similarity(
    user_playlist_title: str,
    n_playlists_to_pull: int,
    similarity_metric: Callable,
    playlist_titles_df: str = "playlist_titles.pickle",
):
    """
    Looks up a DataFrame by title similarity and returns the top N playlists.
    Use this function, when neither the songs nor the title of the user's playlist are known by our embeddings models.
    Args:
        user_playlist_title (str): The title of the user's playlist.
        n_playlists_to_pull (int): The number of playlists to retrieve.
        similarity_metric (Callable): A function that calculates the similarity between two titles.

    Returns:
        pd.DataFrame: The top N playlists sorted by similarity.

    """

    df = pd.read_pickle(playlist_titles_df).to_frame(
        name="title"
    )  # Load all the playlist titles for similarity retrieval...

    df["similarity"] = (
        df["title"]
        .astype(str)
        .apply(lambda title: similarity_metric(user_playlist_title, title))
    )
    df, sim = df.sort_values(by="similarity", ascending=False).head(
        n_playlists_to_pull
    ), df.pop("similarity")
    return list(sim), df.index


def complete_with_playlists(
    seeds: List[str],
    lookup_df: pd.DataFrame,
    strategy: str,
    K: int = 500,
):
    """
    Complete the playlist by filling it up with songs from the most popular or most similar playlists.

    Args:
        seeds (List[str]): List of seed songs to start the playlist.
        lookup_df (pd.DataFrame): DataFrame containing information about playlists and songs.
        strategy (str): Strategy for completing the playlist. Can be "most_popular_playlists" or "most_similar_playlists".
        K (int, optional): Number of songs to complete the playlist. Defaults to 500.

    Returns:
        List[str]: The completed playlist as a list of song titles.
    """
    if strategy not in ["most_popular_playlists", "most_similar_playlists"]:
        raise ValueError(
            f"Invalid strategy {strategy}. Must be 'most_popular_playlists' or 'most_similar_playlists'."
        )
    stop = False
    if strategy == "most_popular_playlists":
        lookup_df = lookup_df.sort_values("popularity", ascending=False)
    playlist_completion = []
    for playlist in lookup_df["songs"]:
        for song in playlist:
            if song not in set(playlist_completion).union(set(seeds)):
                playlist_completion.append(song)
            if len(playlist_completion) == K:
                stop = True
                break
        if stop:
            break
    return playlist_completion


def get_most_n_similar_artists(
    songs: List[str],
    songs_embeddings_model: Word2VecSongsEmbeddings,
    n_most_similar_artists_to_consider: int,
):
    """
    Get the most similar artists to the seed songs.

    This function takes a list of seed songs, a songs embeddings model, and the number of most similar artists to consider.
    It returns a set of the most similar artists to the seed songs based on the songs embeddings model.

    Parameters:
        songs (List[str]): A list of seed songs.
        songs_embeddings_model (Word2VecSongsEmbeddings): The songs embeddings model used to calculate artist similarities.
        n_most_similar_artists_to_consider (int): The number of most similar artists to consider.

    Returns:
        set: A set of the most similar artists to the seed songs.

    Notes:
        - If the playlist is empty, an empty set is returned.
        - The function leverages the relevancy of the songs embeddings model to propose similar artists based on the seed songs.
        - The function considers only the artists that are present in the songs embeddings model.
        - The original artists from the seed songs are also included in the set of most similar artists.
    """
    if not any(songs):  # If the playlist is empty
        return set()
    most_similar_artists = set()
    for artist in set(songs).intersection(songs_embeddings_model.model.wv.index_to_key):
        similar_artists = [
            artist
            for artist, _ in songs_embeddings_model.model.wv.most_similar(
                artist, topn=n_most_similar_artists_to_consider
            )
        ]
        most_similar_artists.update(similar_artists)
    if not most_similar_artists:
        return set()
    # Add original artists
    most_similar_artists.update(set(songs))
    return most_similar_artists


def complete_with_most_popular_songs(
    seeds: List[str],
    songs: np.ndarray,
    lookup_df: pd.DataFrame,
    songs_embeddings_model: Word2VecSongsEmbeddings,
    sim: np.ndarray,
    weighted: bool,
    n_most_sim_artists_to_consider: int = 100,
    K: int = 500,
    filter_by_artists: bool = True,
) -> List[str]:
    """
    Complete the playlist by filling it up with the most popular songs.

    Args:
        seeds (List[str]): Songs of the playlist we want to complete.
        songs (np.ndarray): Array of song titles to the granularity of config file, artists level by default.
        lookup_df (pd.DataFrame): DataFrame containing most similar playlists pulled up previously by similarity.
        songs_embeddings_model (Word2VecSongsEmbeddings): Word2Vec model for songs embeddings.
        sim (np.ndarray): Array of similarity scores.
        weighted (bool): Flag indicating whether to use weighted completion or not.
        n_most_sim_artists_to_consider (int, optional): Number of most similar artists to consider. Defaults to 100.
        K (int, optional): Number of songs to complete the playlist. Defaults to 500.
        filter_by_artists (bool, optional): Flag indicating whether to filter songs by artists or not. Defaults to True.

    Returns:
        List[str]: The completed playlist as a list of song titles.
    """
    recommended_songs = list(lookup_df["songs"])
    allowed_artists = get_most_n_similar_artists(
        songs, songs_embeddings_model, n_most_sim_artists_to_consider
    )
    counter = Counter()
    recommended_songs = [
        (song, sim[i])
        for i, playlist in enumerate(recommended_songs)
        for song in playlist
        if song not in seeds
    ]  # Get the same granularity the model was trained on.
    if (
        filter_by_artists and allowed_artists
    ):  # Filter by artists (if allowed_artists is empty, it means that the user playlist is empty...)
        recommended_songs = [
            (song, sim)
            for song, sim in recommended_songs
            if song.split(" - ")[0] in allowed_artists
        ]  # Remove songs whose artists not belong to the most similar artists of the playlist ones.
    for song, sim in recommended_songs:
        if weighted:
            counter[
                song
            ] += sim  # Update by similarity playlist, the more similar the playlist, the more weight the song is given.
        else:
            counter[song] += 1  # Give the same weight to each song.
    playlist_completion = [song for song, _ in counter.most_common(K)]
    return playlist_completion


def complete_playlist(
    seeds: List[str],
    songs: np.ndarray,
    lookup_df: pd.DataFrame,
    songs_embeddings_model: Word2VecSongsEmbeddings,
    sim: np.ndarray,
    K: int,
    strategy: str,
    n_most_sim_artists_to_consider: int = 100,
    weighted: bool = True,
    filter_by_artists: bool = True,
):
    """
    Complete the playlist by filling it up with recommended songs based on the given seeds and strategy.
    """
    if strategy == "most_popular_songs":
        playlist_completion = complete_with_most_popular_songs(
            seeds,
            songs,
            lookup_df,
            songs_embeddings_model,
            sim,
            weighted,
            n_most_sim_artists_to_consider,
            K,
            filter_by_artists,
        )
    else:
        playlist_completion = complete_with_playlists(seeds, lookup_df, strategy, K)
    return playlist_completion


def next_K_songs(
    seeds: List[str],
    playlist_title: str,
    songs: List[str],
    faiss_index,
    title_embeddings_model: Word2VecTitlesEmbeddings,
    songs_embeddings_model: Word2VecSongsEmbeddings,
    K: int = 500,
    n_playlists_to_pull: int = 100,
    strategy: str = "most_popular_songs",
    pool_method: str = "median",
    weighted: bool = True,
    n_most_sim_artists_to_consider: int = 100,
    filter_by_artists: bool = True,
):
    """
    Completes a playlist by filling it up with recommended songs based on the given seeds and strategy.

    Args:
        seeds (List[str]): List of seed songs to start the playlist.
        playlist_title (str): Title of the playlist.
        songs (List[str]): List of all available songs.
        faiss_index (faiss.swigfaiss.Index): Faiss index used for similarity search.
        title_embeddings_model (Word2VecTitlesEmbeddings): Word2Vec model for title embeddings.
        songs_embeddings_model (Word2VecSongsEmbeddings): Word2Vec model for song embeddings.
        K (int, optional): Number of songs to complete the playlist. Defaults to 500.
        n_playlists_to_pull (int, optional): Number of similar playlists to consider. Defaults to 100.
        strategy (str, optional): Strategy for completing the playlist. Can be "most_similar_songs",
            "most_popular_playlists", "most_similar_playlists", or "most_popular_songs".
            Defaults to "most_similar_songs".

    Returns:
        Tuple[List[str], pd.DataFrame]: A tuple containing the completed playlist as a list of song titles
        and the DataFrame containing information about the recommended playlists.
    """

    user_playlist_embedding = get_user_playlist_embedding(
        playlist_title,
        songs,
        title_embeddings_model,
        songs_embeddings_model,
        pool_method,
    )

    sim, lookup_df = get_playlist_recommendations(
        user_playlist_embedding,
        playlist_title,
        faiss_index,
        n_playlists_to_pull,
    )

    playlist_completion = complete_playlist(
        seeds,
        songs,
        lookup_df,
        songs_embeddings_model,
        sim,
        K,
        strategy,
        n_most_sim_artists_to_consider,
        weighted,
        filter_by_artists,
    )
    return playlist_completion, lookup_df


def create_faiss_index(embeddings: np.ndarray, save: bool = True):
    index = faiss.IndexFlatIP(embeddings.shape[-1])  # Embeddings's shape.
    index.add(embeddings)
    if save:
        faiss.write_index(index, "embeddings_faiss.index")
    return index


def query_embeddings_by_index(
    faiss_index: faiss.Index, index: List[int], batch_size: int = 1000
) -> np.ndarray:
    results = []
    for i in range(0, len(index), batch_size):
        batch_indices = index[i : i + batch_size]
        batch_results = faiss_index.reconstruct_batch(batch_indices)
        results.append(batch_results)
    # Concatenate all batches
    return np.vstack(results)
