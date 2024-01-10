# Sci-Fi Novel Search Engine

## Description

Are you a sci-fi fan who can't figure out what to read next? Then the Sci-Fi Novel Search Engine is for you!
Use the search bar to choose an emotion and a topic - the search engine will show you the most relevant books in its database. 

![App screenshot](image/screenshot.png)

This project is built on the following:
* Dataset: The excellent [SFGram dataset](https://github.com/nschaetti/SFGram-dataset)
* Text pre-processing: [spaCy](https://spacy.io/)
* Emotion classification: DistilBERT tuned on a Twitter emotion dataset by bhadresh-savani, available [here](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
* Topic modeling: BERTopic pretrained on Wikipedia by Maarten Grootendorst, available [here](https://huggingface.co/MaartenGr/BERTopic_Wikipedia)
* Frontend: [Plotly Dash](https://dash.plotly.com/)

## How to run the app
Clone the repository.
```
git clone https://github.com/divijmishra/scifi_nlp_search_engine.git
```
Install dependencies in your environment (Conda env recommended) (note that these dependencies are only sufficient to run the Dash app)
```
pip install -r requirements.txt
```
Run the app:
```
python3 src/app.py
```
You can use the app on your browser at [http://127.0.0.1:8050](http://127.0.0.1:8050).

## How to run the analysis
Clone the repository.
Clone the repository.
```
git clone https://github.com/divijmishra/scifi_nlp_search_engine.git
```
Install dependencies in your environment (Conda env recommended)
```
pip install -r requirements-full.txt
```
Run the scripts in the following order:
```
python3 src/01-setup_directories_and_download_data.py
python3 src/02-model_inference.py
python3 src/03-post-inference_processing.py
```
Note that 02-model_inference.py is likely to take a lot of time. If you like, you can change 
```
for id in ids[0:]:
```
in lines 34 and 98 to, for e.g.,
```
for id in ids[0:10]:
```
to perform model inference for a few books to explore the analysis and scale it up later as per your preference.

## Contributors

Divij Mishra ([divijmishra@gmail.com](divijmishra@gmail.com))

## Acknowledgments

* [The SFGram dataset](https://github.com/nschaetti/SFGram-dataset)
* [spaCy](https://spacy.io/)
* [BERT](https://arxiv.org/abs/1810.04805v2)
* [Emotion classifier](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
* [BERTopic](https://maartengr.github.io/BERTopic/index.html)
* [BERTopic pretrained on Wikipedia corpus](https://huggingface.co/MaartenGr/BERTopic_Wikipedia)
* [An excellent primer on data science project organization](https://drivendata.github.io/cookiecutter-data-science/)
