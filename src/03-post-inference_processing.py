"""
Script to do some post-inference processing 
to make data usable by our frontend.
"""
from bertopic import BERTopic
import pandas as pd
import json
import os

##### in case some books weren't handled correctly
##### by either the emotion detector or the topic
##### modeller, we'll remove them

emotion_path = "data/interim/emotions/"
topic_path = "data/interim/topics/"

em_ids, top_ids = [], []

for f in os.listdir(emotion_path):
    if f.endswith("txt"):
        em_ids.append(f[:3])

for f in os.listdir(topic_path):
    if f.endswith("csv"):
        top_ids.append(f[:3])

em_set, top_set = set(em_ids), set(top_ids)

bad_em_ids = []
for id in em_ids:
    if id not in top_set:
        bad_em_ids.append(id)

bad_top_ids = []
for id in top_ids:
    if id not in em_set:
        bad_top_ids.append(id)

for id in bad_em_ids:
    f = emotion_path + id + ".txt"
    if os.path.exists(f):
        os.remove(f)

for id in bad_top_ids:
    f = topic_path + id + ".csv"
    if os.path.exists(f):
        os.remove(f)


##### collect emotions data
load_path = "data/interim/emotions/"
save_path = "data/final/"

emotions = {
    "joy": {},
    "love": {},
    "anger": {},
    "sadness": {},
    "fear": {},
    "surprise": {},
}

ids = []
for f in os.listdir(load_path):
    ids.append(f[:3])
ids.sort()

for id in ids:
    f = id + ".txt"
    with open(load_path + f) as file:
        ems = json.load(file)

    # normalize emotion scores
    tot = sum([ems[em] for em in ems])
    for em in ems:
        ems[em] /= tot

    # save them to the dict
    for em in emotions:
        emotions[em][id] = ems[em]

# write emotion dict
out_file = save_path + "emotions.json"
with open(out_file, "w") as file:
    json.dump(emotions, file)


##### 2 tasks:
##### 1) Gather topic_ids in a dict, with each topic_id's value being another dict.
#####    The inner dict contains that topic_id's score in each book.
##### 2) Make a dict of all words used in these topics, with the values being their topic_id.

load_path = "data/interim/topics/"
save_path = "data/final/"

# get BERTopic model info
model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
df_fulltopic = model.get_topic_info()
df = df_fulltopic[["Topic", "Representation"]]
df = df.drop([0])  # remove topic -1

full_vocab_dict = {}
full_vocab_dict_reversed = {}

for i in df.index:
    topic_id, words = df["Topic"][i], tuple(df["Representation"][i])
    full_vocab_dict[topic_id] = words

for k, v in full_vocab_dict.items():
    full_vocab_dict_reversed[v] = k

# Task 1

ids = []
for f in os.listdir(load_path):
    if f.endswith(".csv"):
        ids.append(f[:3])
ids.sort()

topic_dict = {}

for id in ids:
    f = id + ".csv"
    df_book = pd.read_csv(load_path + f)

    # normalize probabilities
    tot = sum(df_book["prob"])
    df_book["prob"] /= tot

    try:
        for i in df_book.index:
            p, words = df_book["prob"][i], tuple(eval(df_book["words"][i])[0])
            topic_id = full_vocab_dict_reversed[words]

            topic_dict.setdefault(str(topic_id), {})[id] = p

    except:
        # super weird, one topic is present in book topics
        # but I can't find it in BERTopic_wikipedia's topic info
        # even though that's the model that was used to generate
        # the book topics. no clue why.
        pass
        # print(words)
        # print(id)

# write topics dict
out_file = save_path + "topics.json"
with open(out_file, "w") as file:
    json.dump(topic_dict, file)

# Task 2

vocab = {}

for topic_id in topic_dict:
    words = full_vocab_dict[int(topic_id)]
    for word in words:
        vocab.setdefault(word, []).append(topic_id)

# write vocab mapping file
out_file = save_path + "vocab.json"
with open(out_file, "w") as file:
    json.dump(vocab, file)
