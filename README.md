<div align="center">
  <img src="app/assets/logo.png" alt="Project Logo" width="250"/>
</div>

Your Personalized Music Companion

Discover, explore, and play your favorite tracks with LucileJams. Trained on the Spotify Million Playlist Dataset, LucileJams offers a sophisticated recommendation pipeline built from scratch. Simply input a title and artists you love, and LucileJams will provide you with the next best songs to listen to. Inspired by the legendary guitar of B.B. King, LucileJams combines elegance and functionality to bring you an unparalleled musical experience. Tune in and let LucileJams be the soundtrack to your life.


## Project Demo

Check out this demo of my music recommendation system.

<div align="center">
<a href="https://www.youtube.com/watch?v=U9KajAO0UXk">
  <img src="https://img.youtube.com/vi/U9KajAO0UXk/maxresdefault.jpg" alt="Watch the demo" width="400"/>
</a>
</div>

## Workflow

### Overview

What can we do with user-generated playlists, and how can we create an algorithm from scratch to automate recommendations for other users? The core idea driving this project's architecture is to map playlists in a high-dimensional space and then use similarity to generate recommendations. 

The main challenge then becomes: **How can we create meaningful embeddings from the training set available?**

To briefly explain, the **Spotify Million Playlist Dataset** consists of user-created playlists made up of songs, where each song is represented by its album, title, and artist. No additional metadata, such as lyrics, was provided.

Inspired by the classical Word2Vec architecture, which predicts a word based on its neighbors, we aim to apply similar principles here. Word2Vec has historically shown great success in generating meaningful embeddings.

The following diagrams explain the structure of the project. As the saying goes, "a picture is worth a thousand words."

## How is the model trained?

<div align="center">
  <img src="./workflow/training_pipeline.svg" alt="Training Pipeline" width="600"/>
</div>

## How is a playlist embedding created?

<div align="center">
  <img src="./workflow/playlist_embedding_logic.svg" alt="Playlist Embedding Logic" width="600"/>
</div>

## How do all the components come together to generate recommendations and find similar playlists?

<div align="center">
  <img src="./workflow/inference_pipeline.svg" alt="Inference Pipeline" width="800"/>
</div>


## Instructions 

### Prerequisites

Before running the app, make sure you have the following installed:

Make sure you have the following installed:
- **Python 3.x** (You can check your Python version by running `python --version` or `python3 --version`)
- **pip** (Python package installer)
- **venv** (Virtual environment module, which is included by default with Python 3.x. On some Linux systems, it may need to be installed separately using the package manager: sudo apt install python3-venv).


### 1. Clone the Repository

To download the app, clone the repository from GitHub using the following command:

```bash
git clone https://github.com/tomargentin0901/LucilleJams
```
Navigate into the project directory : 

```bash
cd LucilleJams
```

### 2. Set Up a Virtual Environment and Install Required Dependencies

To avoid conflicts with other Python projects, it is recommended to create a virtual environment before installing the dependencies. 

For **macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

For **Windows (using cmd)**:

```bash
python -m venv venv
venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download, Merge and Unzip Required Data Files

Before running the app, download the required files from the GitHub release. Since the FAISS index is split into multiple parts, you need to download all parts and then merge them.

#### Download the Files:

1. [FAISS Index Part 1](https://github.com/tomargentin0901/LucilleJams/releases/download/1.0.0/faiss.index.part-aa)
2. [FAISS Index Part 2](https://github.com/tomargentin0901/LucilleJams/releases/download/1.0.0/faiss.index.part-ab)
3. [Spotify Dataset](https://github.com/tomargentin0901/LucilleJams/releases/download/1.0.0/spotify_dataset.gz)

**Once the download is complete, move the downloaded files into the GitHub repository folder.**

### 4. Prepare the FAISS Index and Spotify Dataset :

The repository includes several FAISS index parts and the Spotify dataset in a compressed format. 
Instead of manually merging the index parts and unzipping the dataset, you can run the provided Python script to automate this process.

Simply run the following command after setting up the environment:

```bash
python process_files.py
```

This script will:

- Merge the FAISS Index Parts: It will combine the downloaded faiss.index.part-* files into a single embeddings_faiss.index file and remove the part files.
- Unzip the Spotify Dataset: It will unzip spotify_dataset.gz and rename it to spotify_dataset.db automatically.
  

Your project folder should look like this:

```plaintext
LucilleJams/
│
├── app.py
├── faiss.index
├── your-pickle-file.pkl
├── spotify_dataset.db
├── process_files.py       
├── requirements.txt
├── README.md
├── config/
│   ├── config_word2vec.py
│   ├── emoji_name_to_text.py
│   ├── tokenizer_config.py
├── models/
│   ├── embeddings.py
│   ├── songs_model.pickle
│   ├── title_model.pickle
└── utils/
    ├── dataset_utils.py
    ├── inference_utils.py
    ├── visualization_utils.py


```

### 4. Run the app 

Once all files are in place, you can run the app using the following command:

```bash
python -m app.app
```

After running the command, open your web browser and navigate to [http://127.0.0.1:8050/](http://127.0.0.1:8050) to access the application.

---



