import os
import glob
import shutil
import gzip


# For unzipping spotify_dataset.gz and renaming it to spotify_dataset.db
def unzip_and_rename():
    with gzip.open("spotify_dataset.gz", "rb") as f_in:
        with open("spotify_dataset.db", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove("spotify_dataset.gz")


# Function to concatenate faiss index parts and remove them
def concatenate_and_remove():
    # Find all files that match the pattern 'faiss.index.part-*'
    part_files = sorted(glob.glob("faiss.index.part-*"))

    # Concatenate all the part files into 'embeddings_faiss.index'
    with open("embeddings_faiss.index", "wb") as outfile:
        for part_file in part_files:
            with open(part_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    # Remove the part files after concatenation
    for part_file in part_files:
        os.remove(part_file)


# Main script execution
if __name__ == "__main__":
    try:
        # Unzip spotify_dataset.gz and rename to spotify_dataset.db
        unzip_and_rename()
    except FileNotFoundError:
        print("spotify_dataset.gz not found")

    try:
        # Concatenate faiss index parts and remove them
        concatenate_and_remove()
    except FileNotFoundError:
        print("faiss.index.part-* files not found")
