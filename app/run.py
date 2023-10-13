import json
import plotly
import pandas as pd
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals 
import joblib
import sqlalchemy
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from PIL import Image
import tempfile


# python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

app = Flask(__name__,static_folder='static')

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This custom transformer class extracts the starting verb of a sentence
    and creates a new feature for a machine learning classifier.
    
    Attributes:
        None

    Methods:
        starting_verb(self, text):
            Extracts the starting verb of a sentence.

        fit(self, X, y=None):
            Fits the transformer on the input data.
            
        transform(self, X):
            Transforms the input data by adding a binary feature indicating
            whether the sentence starts with a verb.

    Example:
        # Create a StartingVerbExtractor instance
        verb_extractor = StartingVerbExtractor()
        
        # Fit and transform the data
        X_new = verb_extractor.transform(X)
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
# Define performance metric for use in grid search scoring object
def grid_eval_metric(y_true, y_pred):
    """Calculate median gmean score for all of the output classifiers
    
    Args:
    y_true: array. Array containing actual labels.
    y_pred: array. Array containing predicted labels.
        
    Returns:
    score: float. Median gmean score for all of the output classifiers
    """
    gmean_list = []
    
    for i in range(np.shape(y_pred)[1]):
        accuracy = accuracy_score(np.array(y_true)[:, i], y_pred[:, i])
        recall = recall_score(np.array(y_true)[:, i], y_pred[:, i],zero_division=1)
        gmean = 2 * (recall * accuracy) / (recall + accuracy)
        gmean_list.append(gmean)
       
        
    score = np.median(gmean_list)
    print(f'score: {score}')
    return score

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# Function to generate a word cloud image
def generate_word_cloud(text_data):
    # Generate the word cloud image
    wordcloud = WordCloud(width=800, height=400).generate(text_data)

    # Create a temporary file to save the word cloud image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image_path = temp_file.name
        wordcloud.to_file(temp_file.name)

    return image_path


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # category_names = df.iloc[:,4:].columns
    # category_boolean = (df.iloc[:,4:] != 0).sum().values

    categorical_columns = df.iloc[:,4:].columns
    dfcat=df.iloc[:,4:].reset_index()
    dfcat.head(2)
    dfcat=dfcat.drop(columns=['index'])
    counts = dfcat.sum().sort_values()

    category_names = counts.index
    category_boolean = counts.values


    
    # # Sample text data (replace with your own text data)
    # text_data = "This is a simple word cloud example. " \
    #             "You can replace this text with your own data."

    # # Generate the word cloud image
    # word_cloud_image = generate_word_cloud(text_data)

    # return render_template('master.html', word_cloud_image=word_cloud_image)


    # Define custom colors for the bars
    genre_bar_colors = ['teal', 'darkblue', 'green', 'red', 'orange']
    cat_bar_colors = ['teal', 'darkblue', 'green', 'red', 'orange']
    category_bar_color = 'rgb(255, 128, 0)'
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color=genre_bar_colors)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'paper_bgcolor': 'rgba(0,0,0,0)',  # Make the background transparent
                'plot_bgcolor': 'rgba(0,0,0,0)' 
            }
        },
         # GRAPH 2 - category graph    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    # ,
                    marker=dict(color='steelblue')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count of messages"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45,
                    'titlepad': 20 
                }
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # text_data = "This is a simple word cloud example. " 
    # word_cloud_image_path = generate_word_cloud(text_data)

    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()