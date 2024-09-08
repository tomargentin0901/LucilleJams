import pandas as pd
from collections import Counter
import numpy as np

import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from utils.dataset_utils import query_data_by_index
from utils.inference_utils import query_embeddings_by_index
from sklearn.cluster import KMeans
import faiss
from models.embeddings import Word2VecTitlesEmbeddings

import nltk
from nltk.corpus import stopwords
import nltk
import string


# Function to check and install NLTK data
def check_and_install_nltk_data():
    try:
        # Check if 'stopwords' is already downloaded
        nltk.data.find("corpora/stopwords")
    except LookupError:
        # Download if not found
        nltk.download("stopwords")


# Run the function to ensure NLTK data is available
check_and_install_nltk_data()


def get_most_K_frequent_artists(
    df: pd.DataFrame,
    songs_col_name: str = "songs",
    K: int = 3,
    output_col_name: str = "most_frequent_artists",
) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        songs_col_name (str, optional): _description_. Defaults to "songs".
        K (int, optional): _description_. Defaults to 3.
        output_col_name (str, optional): _description_. Defaults to "most_frequent_artists".

    Returns:
        pd.DataFrame: _description_
    """
    list_of_songs = df[songs_col_name].values
    list_of_artists = [
        list(map(lambda song: song.split(" - ")[0], songs)) for songs in list_of_songs
    ]  # Get the artist only
    most_common_artists = [
        " - ".join(list(dict(Counter(artist).most_common(3)).keys()))
        for artist in list_of_artists
    ]  # Get the top K most frequent artists for each playlist
    df.loc[:, output_col_name] = most_common_artists
    return df


def get_playlist_label_to_hover(
    df: pd.DataFrame,
    most_freq_artists_col_name: str = "most_frequent_artists",
    output_col_name: str = "to_hover",
) -> pd.Series:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        most_freq_artists_col_name (str, optional): _description_. Defaults to "most_frequent_artists".
        output_col_name (str, optional): _description_. Defaults to "to_hover".

    Returns:
        pd.Series: _description_
    """
    df = get_most_K_frequent_artists(df)
    df.loc[:, output_col_name] = (
        "PLAYLIST TITLE : "
        + df["title"]
        + " <br> "
        + "TOP ARTISTS : "
        + df[most_freq_artists_col_name]
    )
    return df[output_col_name].values


def get_centroid_playlist_title(
    df: pd.DataFrame,
    num_cluster: int,
    title_embeddings_model: Word2VecTitlesEmbeddings,
    playlist_title_colname: str = "playlist_title",
    stop_words=set(stopwords.words("english")),
) -> np.ndarray:

    # Create a translation table for removing punctuation
    translator = str.maketrans("", "", string.punctuation)

    df[playlist_title_colname] = df[playlist_title_colname].apply(
        lambda title: (
            " ".join(
                [
                    word
                    for word in title.translate(translator).split()
                    if word.lower() not in stop_words
                ]
            )
            if title
            else ""
        )
    )  # Remove noise for embeddings...

    cluster_playlist_titles = list(
        df[df["cluster"] == num_cluster][playlist_title_colname].values
    )

    cluster_playlist_titles = list(
        filter(None, cluster_playlist_titles)
    )  # remove playlists without title
    title_embeddings = title_embeddings_model.get_titles_embeddings(
        cluster_playlist_titles, title_embeddings_model.model
    )
    playlist_title_centroid = np.mean(title_embeddings, axis=0)

    return playlist_title_centroid


def get_island_title(
    df: pd.DataFrame,
    num_cluster: int,
    title_embeddings_model: Word2VecTitlesEmbeddings,
    playlist_title_colname: str = "playlist_title",
    stop_words=set(stopwords.words("english")),
):
    centroid_playlist_title = get_centroid_playlist_title(
        df, num_cluster, title_embeddings_model, playlist_title_colname, stop_words
    )
    suggested_titles = title_embeddings_model.model.similar_by_vector(
        centroid_playlist_title, topn=20
    )
    return suggested_titles


def map_playlists_in_subspace(
    index: faiss.Index,
    user_playlist_embedding: np.ndarray,
    out_dim: int = 2,
    N: int = 10000,
    perplexity: int = 10,
):
    """Compute t-SNE embeddings for playlists and user playlist."""
    random_index = np.random.choice(index.ntotal, size=(N,), replace=False)
    random_index = sorted([int(idx_) for idx_ in list(random_index)])
    df = query_data_by_index(random_index)

    # Query embeddings
    all_embeddings = np.vstack(
        [query_embeddings_by_index(index, random_index), user_playlist_embedding]
    )

    # Compute t-SNE
    tsne = TSNE(n_components=out_dim, perplexity=perplexity)
    embeddings_tsne = tsne.fit_transform(all_embeddings)

    return embeddings_tsne, df


def cluster_playlists(embeddings_tsne: np.ndarray, n_clusters: int = 20):
    """Perform KMeans clustering on t-SNE reduced points."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_tsne)
    cluster_labels[-1] = -1  # Custom marker for the user playlist
    return cluster_labels, kmeans


def generate_cluster_titles(
    mapped_df: pd.DataFrame,
    title_embeddings_model: Word2VecTitlesEmbeddings,
    playlist_title_colname: str = "playlist_title",
):
    """Generate cluster titles using the provided embeddings model."""
    cluster_centroids = mapped_df.groupby("cluster")[["x", "y"]].mean().reset_index()

    cluster_titles = {}
    for cluster in cluster_centroids["cluster"]:
        if cluster != -1:  # Skip the user playlist cluster
            suggested_titles = get_island_title(
                mapped_df, cluster, title_embeddings_model, playlist_title_colname
            )
            cluster_titles[cluster] = suggested_titles[0][0]

    cluster_centroids["title"] = cluster_centroids["cluster"].map(cluster_titles)

    return cluster_titles, cluster_centroids


def create_scatter_plot(
    mapped_df: pd.DataFrame,
    cluster_centroids: pd.DataFrame,
    user_playlist_cluster_title: str,
):
    """Create the scatter plot with cluster annotations."""
    fig = px.scatter(
        mapped_df,
        x="x",
        y="y",
        hover_name="label",
        title=f"Welcome to the Island of {user_playlist_cluster_title}!<br>Navigate through your musical paradise, explore the neighboring lands of sound, and uncover new treasures waiting to be discovered!",
        labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
        color="color",
        size="size",
        hover_data={"x": False, "y": False, "color": False, "size": False},
    )
    # Center the title
    fig.update_layout(title_x=0.5)
    # Add text annotations for each cluster centroid
    for _, row in cluster_centroids.iterrows():
        if row["cluster"] != -1:  # Skip the user playlist cluster
            fig.add_annotation(
                x=row["x"],
                y=row["y"],
                text=row["title"],
                showarrow=False,
                font=dict(size=16, color="white"),  # Larger font size and white color
                bgcolor="rgba(0, 0, 0, 0.6)",  # Black background with transparency
                align="center",
            )

    # Customize user playlist point
    user_point = mapped_df[mapped_df["cluster"] == -1]
    fig.add_trace(
        px.scatter(
            user_point,
            x="x",
            y="y",
            hover_name="label",
            color_discrete_sequence=["red"],
            size="size",
        ).data[0]
    )

    return fig


def get_neighbors_plot(
    index: faiss.Index,
    user_playlist_embedding: np.ndarray,
    title_embeddings_model: Word2VecTitlesEmbeddings,
    playlist_title_colname: str = "playlist_title",
    playlist_title: str = "USER PLAYLIST",
    out_dim: int = 2,
    N: int = 10000,
    perplexity: int = 10,
    n_clusters: int = 20,
):
    # Step 1: Compute t-SNE embeddings
    embeddings_tsne, df = map_playlists_in_subspace(
        index, user_playlist_embedding, out_dim, N, perplexity
    )

    # Step 2: Perform KMeans clustering
    cluster_labels, kmeans = cluster_playlists(embeddings_tsne, n_clusters)

    # Step 2.1: Determine the closest cluster for the user playlist embedding
    user_embedding_tsne = embeddings_tsne[-1]
    cluster_centroids = kmeans.cluster_centers_
    user_cluster_label = np.argmin(
        np.linalg.norm(cluster_centroids - user_embedding_tsne, axis=1)
    )

    # Step 3: Prepare DataFrame for plotting
    hovering_label = list(get_playlist_label_to_hover(df)) + [playlist_title]
    mapped_df = pd.DataFrame(
        {
            "x": embeddings_tsne[:, 0],
            "y": embeddings_tsne[:, 1],
            "label": hovering_label,
            "cluster": cluster_labels,
            "size": [5] * (len(hovering_label) - 1)
            + [20],  # Make the last point even bigger
            "playlist_title": list(df["title"].values) + [playlist_title],
        }
    )
    mapped_df["color"] = mapped_df["cluster"].apply(
        lambda x: "red" if x == -1 else f"cluster_{x}"
    )

    # Step 4: Generate titles for each cluster
    cluster_titles, cluster_centroids = generate_cluster_titles(
        mapped_df, title_embeddings_model, playlist_title_colname
    )
    user_cluster_label = cluster_titles[user_cluster_label]

    # Step 5: Create and display the scatter plot
    fig = create_scatter_plot(mapped_df, cluster_centroids, user_cluster_label)
    return fig
