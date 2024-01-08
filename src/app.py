# import packages
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os


##### data handling without viz


# load data
load_path = "data/final/"

with open(load_path + "emotions.json") as f:
    emotions = json.load(f)
with open(load_path + "topics.json") as f:
    topics = json.load(f)
with open(load_path + "vocab.json") as f:
    vocab = json.load(f)
with open(load_path + "books.json") as f:
    books = json.load(f)

emotion_searchbar_list = list(emotions.keys())
emotion_searchbar_list.sort()
emotion_searchbar_list.insert(0, "<no_selection>")
topic_word_searchbar_list = list(vocab.keys())
topic_word_searchbar_list.sort()
topic_word_searchbar_list.insert(0, "<no_selection>")
book_searchbar_list = [v["title"] for k, v in books.items()]
book_searchbar_list.sort()
book_name_dict = {books[id]["title"]: id for id in books}


# functions to process data
def query_1(emotion, word):
    """
    given an emotion and a topic word,
    return books sorted in descending order wrt
    the product of their emotion and topic word scores.
    """

    if emotion == "<no_selection>" and word == "<no_selection>":
        # figure out placeholder
        return None

    if emotion == "<no_selection>":
        # only need to sort by word scores
        relevant_topics = vocab[word]

        word_scores_dict_list = [topics[topic] for topic in relevant_topics]
        word_score_dict = {}
        for d in word_scores_dict_list:
            for k in d:
                word_score_dict[k] = word_score_dict.get(k, 0) + d[k]

        final_dict = {books[id]["title"]: word_score_dict[id] for id in word_score_dict}
        return final_dict

    if word == "<no_selection>":
        # only need to sort by emotion scores
        emotion_dict = emotions[emotion]
        final_dict = {books[id]["title"]: emotion_dict[id] for id in emotion_dict}
        return final_dict

    # else emotions and topic words
    emotions_dict, relevant_topics = emotions[emotion], vocab[word]

    word_scores_dict_list = [topics[topic] for topic in relevant_topics]
    word_score_dict = {}
    for d in word_scores_dict_list:
        for k in d:
            word_score_dict[k] = word_score_dict.get(k, 0) + d[k]

    # product of the two
    final_dict = {}
    for id in word_score_dict:
        book_name = books[id]["title"]
        final_dict[book_name] = word_score_dict[id] * emotions_dict[id]
    return final_dict


##### visualization


# initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                html.H1(children="The Sci-Fi book search engine!", className="header"),
                html.Hr(),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            children="""
                            Select an emotion and/or a topic word to see the most 
                            relevant books.
                            """
                        ),
                        dcc.Dropdown(
                            emotion_searchbar_list,
                            emotion_searchbar_list[0],
                            id="emotion-dropdown",
                            # placeholder="Choose an emotion from the dropdown."
                        ),
                        dcc.Dropdown(
                            topic_word_searchbar_list,
                            topic_word_searchbar_list[0],
                            id="topic-word-dropdown",
                            # placeholder="Start typing your word of interest."
                        ),
                        dcc.Graph(figure={}, id="graph1"),
                    ],
                    # width=6,
                ),
                dbc.Col(
                    [
                        html.H3(
                            children="""
                            Select a book to see the most significant topics in that 
                            book."""
                        ),
                        dcc.Dropdown(
                            book_searchbar_list,
                            book_searchbar_list[0],
                            id="book-dropdown",
                        ),
                        html.Div(children="", id="book-information"),
                        dcc.Graph(figure={}, id="graph2"),
                    ],
                    # width=6,
                ),
            ]
        ),
    ],
    fluid=True,
)


##### helper functions for the visualization


# plot 1
@callback(
    Output(component_id="graph1", component_property="figure"),
    Input(component_id="emotion-dropdown", component_property="value"),
    Input(component_id="topic-word-dropdown", component_property="value"),
)
def update_plot1(emotion, topic_word):
    product_dict = query_1(emotion, topic_word)

    if product_dict:
        df = pd.DataFrame([product_dict]).transpose()
        df = df.reset_index()
        df = df.rename(columns={"index": "book_name", 0: "score"})
        df = df.sort_values(by="score", ascending=False)
        df = df.head(10)

        fig = px.bar(df, x="score", y="book_name", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        # placeholder figure
        fig = go.Figure()

        fig.add_annotation(
            go.layout.Annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="No emotion or topic word selected.",
                showarrow=False,
                font=dict(size=20),
            )
        )

        # Update layout to remove axis ticks and labels
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

    return fig


# plot 2
@callback(
    Output(component_id="graph2", component_property="figure"),
    Input(component_id="book-dropdown", component_property="value"),
)
def update_plot2(book_name):
    book_id = book_name_dict[book_name]
    topic_probs_words = books[book_id]["topic_words"]
    df = pd.DataFrame(topic_probs_words)
    df = df.rename(columns={0: "score", 1: "topic_words"})
    df.sort_values(by="score")
    df = df.head(10)
    df["small_topic"] = df["topic_words"].apply(lambda x: x[:75] + " ... ")

    fig = px.bar(df, x="score", y="topic_words", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_yaxes(ticktext=df["small_topic"], tickvals=df["topic_words"])

    return fig


# book information
@callback(
    Output(component_id="book-information", component_property="children"),
    Input(component_id="book-dropdown", component_property="value"),
)
def update_book_info(book_name):
    book_id = book_name_dict[book_name]
    author_name = books[book_id]["author_name"]
    output = f"""
    Author: {books[book_id]["author_name"]}  | 
    Year: {books[book_id]["year"]}    
    """
    return output


##### main function

# run the app
if __name__ == "__main__":
    app.run(debug=True)
