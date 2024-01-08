"""
Executes model inference tasks on the full book corpus:
1) Emotion detection
2) Topic modelling
"""
from transformers import pipeline
from bertopic import BERTopic
import pandas as pd
import spacy
import json


##### emotion detection

# load book metadata dataframe
df_metadata = pd.read_csv("data/interim/books/books_metadata.txt", dtype="str")
ids = list(df_metadata["id"])

failed_ids_sa = []

# directories
load_path = "data/interim/books/fulltexts/"
save_path = "data/interim/emotions/"

# models
nlp = spacy.load("en_core_web_sm")
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    truncation=True,
    top_k=None,
)

for id in ids[0:]:
    try:
        # open the fulltext
        book_path = load_path + f"book{id}.txt"
        with open(book_path) as file:
            fulltext = file.read()

        # split into satisfactory chunks using spaCy
        doc = nlp(fulltext)

        chunk_lists = []
        token_count = 0
        chunk_list = []
        for sent in doc.sents:
            l = len(sent)
            if token_count + l > 400:
                # 400 is an arbitrary value, keeping a large buffer
                # to keep BERT's tokens below 512
                chunk_lists.append(chunk_list)
                token_count = l
                chunk_list = []
                chunk_list.append(sent.text)
            else:
                token_count += l
                chunk_list.append(sent.text)
        chunks = [" ".join(chunk_list) for chunk_list in chunk_lists]

        # run emotion detection on each chunk
        preds = [classifier(chunk) for chunk in chunks]

        # average emotion scores across the document
        emotions = ["joy", "love", "anger", "sadness", "fear", "surprise"]
        scores = [0, 0, 0, 0, 0, 0]
        final_scores = {key: 0 for key in emotions}
        for pred in preds:
            for i in range(6):
                score = pred[0][i]["score"]
                em = pred[0][i]["label"]
                final_scores[em] += score

        # write dict to file
        out_file = save_path + f"{id}.txt"
        with open(out_file, "w") as file:
            json.dump(final_scores, file)
        print(f"ID:{id} done.")

    except:
        failed_ids_sa.append(id)
        print(f"ID:{id} failed.")


##### topic modelling

failed_ids_tm = []

# directories
load_path = "data/interim/books/fulltexts/"
save_path = "data/interim/topics/"

# models
nlp = spacy.load("en_core_web_sm")
model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
df_fulltopic = model.get_topic_info()

for id in ids[0:]:
    try:
        # open the fulltext
        book_path = load_path + f"book{id}.txt"
        with open(book_path) as file:
            fulltext = file.read()

        # split into satisfactory chunks using spaCy
        doc = nlp(fulltext)

        chunk_lists = []
        token_count = 0
        chunk_list = []
        for sent in doc.sents:
            l = len(sent)
            if token_count + l > 400:
                # 400 is an arbitrary value, keeping a large buffer
                # to keep BERT's tokens below 512
                chunk_lists.append(chunk_list)
                token_count = l
                chunk_list = []
                chunk_list.append(sent.text)
            else:
                token_count += l
                chunk_list.append(sent.text)
        chunks = [" ".join(chunk_list) for chunk_list in chunk_lists]

        # run topic modelling inference on each chunk
        topics, probabilities = model.transform(chunks)

        # extract topics and probabililties
        topic_dicts = {}

        for topic in topics:
            topic_dicts[topic] = {
                "prob": 0,
                "words": list(
                    df_fulltopic.loc[df_fulltopic["Topic"] == topic]["Representation"]
                ),
            }

        for i, topic in enumerate(topics):
            topic_dicts[topic]["prob"] += probabilities[i]

        df_topic = pd.DataFrame.from_dict(topic_dicts, orient="index")
        df_topic = df_topic.sort_values(by="prob", ascending=False)

        # write topic_df to file
        out_file = save_path + f"{id}.csv"
        df_topic.to_csv(out_file, index=False)
        print(f"ID:{id} done.")

    except:
        failed_ids_tm.append(id)
        print(f"ID:{id} failed.")
