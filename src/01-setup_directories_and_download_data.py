"""
Creates the required directory structure for the data.
Downloads the SFGram dataset.
Extracts the SFGram dataset and reorganizes some files for next steps.
"""
import pandas as pd
import requests
from tqdm import tqdm
from zipfile import ZipFile
import json
import os


# create directories in data/
os.makedirs("data/raw/", exist_ok=True)
os.makedirs("data/interim/books/fulltexts", exist_ok=True)
os.makedirs("data/interim/emotions/", exist_ok=True)
os.makedirs("data/interim/topics/", exist_ok=True)

# download data
# data_url = "https://databank.worldbank.org/data/download/WDI_CSV.zip"
data_url = "https://github.com/nschaetti/SFGram-dataset/archive/master.zip"
data_path = "data/raw/SFGram-dataset-master.zip"

with requests.get(data_url, stream=True) as response:
    file_size = int(response.headers.get("Content-Length", 0))
    with tqdm(total=file_size, unit="B", unit_scale=True) as progress_bar:
        with open(data_path, mode="wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                progress_bar.update(len(chunk))
                file.write(chunk)

# extract data
extract_path = "data/raw/"
with ZipFile(data_path) as zf:
    for member in tqdm(zf.infolist(), desc="Extracting "):
        try:
            zf.extract(member, extract_path)
        except:
            pass

# extract metadata to a working folder
# load books metadata
load_path = "data/raw/SFGram-dataset-master/books/"

books = []

for f in os.listdir(load_path):
    if f.endswith("json"):
        with open(load_path + f) as file:
            book_data = json.load(file)

        # filter out magazines
        if not (
            book_data["title"].startswith("Galaxy")
            or book_data["title"].startswith("IF")
        ):
            # check if the fulltext actually exists in the data
            id = f[7:10]  # only need 4 digits because << 10k books
            book_fulltext_path = "data/raw/SFGram-dataset-master/book-contents/"
            book_fulltext_path += f"book00{id}.txt"
            if os.path.isfile(book_fulltext_path):
                book = {
                    "id": id,
                    "title": book_data["title"],
                    "author": book_data["author_name"],
                    "year": book_data["year"],
                }

                books.append(book)

df = pd.DataFrame(books)
df = df.sort_values(by="id")
save_path = "data/interim/books/books_metadata.txt"
df.to_csv(save_path, index=False)

# extract book fulltexts to a working folder
df_metadata = pd.read_csv("data/interim/books/books_metadata.txt", dtype="str")
ids = list(df_metadata["id"])

# extract book fulltext, update book metadata dataframe
load_path = "data/raw/SFGram-dataset-master/book-contents/"
save_path = "data/interim/books/fulltexts/"

fulltexts = {}

for id in ids:
    book_load = load_path + f"book00{id}.txt"
    book_save = save_path + f"book{id}.txt"

    with open(book_load, "r") as file:
        fulltext = file.read()

    with open(book_save, "w") as file:
        file.write(fulltext)
