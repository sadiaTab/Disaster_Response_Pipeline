import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import pickle
from sqlalchemy import create_engine, inspect
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    make_scorer,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier

# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# #suppress warnings
# import warnings
# from sklearn.exceptions import DataConversionWarning

# # Ignore specific deprecation warnings
# warnings.filterwarnings("ignore", category=DataConversionWarning)

import warnings

warnings.simplefilter('ignore')

# python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

def load_data(database_filepath):
    """
    Load data from a SQLite database and return the necessary components for building a machine learning model.

    Args:
        database_filepath (str): The file path to the SQLite database containing the data.

    Returns:
        X (pandas.DataFrame): A pandas DataFrame containing the messages.
        y (pandas.DataFrame): A pandas DataFrame containing the target categories.
        category_names (list): A list of category names corresponding to the columns in the y DataFrame.

    Example:
        database_filepath = "DisasterResponse.db"
        X, y, category_names = load_data(database_filepath)
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","")+ "_table"
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)
    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()
    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
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

def build_model():
    """
    Build a machine learning model for multi-output classification using a pipeline.

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn pipeline that includes the following steps:
            - CountVectorizer: Converts text data into a matrix of token counts.
            - TfidfTransformer: Applies TF-IDF (Term Frequency-Inverse Document Frequency) normalization.
            - MultiOutputClassifier with RandomForestClassifier: A multi-output classifier using a Random Forest model.

    Example:
        # Create a machine learning model
        model = build_model()
        
        # Fit the model on training data and make predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    """
    # Define a scikit-learn pipeline for multi-output classification
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    #Define set of hyper-parameters for grid search for hyperparameter tuning
    parameters = {
        'clf__estimator__n_estimators': [10, 20], 
        'clf__estimator__min_samples_split': [2, 5, 10] 
    }

    scorer = make_scorer(grid_eval_metric)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring = scorer)
    
    return cv 

#Method for performance evaluation
def eval_result(y_test, y_pred):
    y_pred_pd = pd.DataFrame(y_pred, columns=y_test.columns)
    res=[]
    col_names = y_test.columns
    for col in col_names:
        accuracy = accuracy_score(y_test[col], y_pred_pd[col])
        precision = precision_score(y_test[col], y_pred_pd[col],zero_division=1)
        recall = recall_score(y_test[col], y_pred_pd[col],zero_division=1)
        gmean = 2 * (recall * accuracy) / (recall + accuracy)
        f1score = f1_score(y_test[col], y_pred_pd[col],zero_division=1)
        res.append([accuracy, precision, recall, gmean, f1score])
        # print(f"{col}, Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, G-Mean: {gmean:.2f}, F1 score: {f1score:.2f}")
    res = np.array(res)
    res_df = pd.DataFrame(data = res, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'G-Mean','F1'])
    return res_df
        
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning model and display the outcomes.

    Args:
        model (object): The trained machine learning model to be evaluated.
        X_test (array-like): The input test data.
        Y_test (array-like): The true labels for the test data.
        category_names (list of str): A list of category names for display.

    Returns:
        None

    Example:
        # Evaluate a machine learning model
        evaluate_model(model, X_test, Y_test, category_names)
    """
    # Make predictions using the provided model
    Y_pred = model.predict(X_test)
    
    #Compute evaluation metrics and show results
    evalres = eval_result(Y_test,Y_pred)
    print(evalres)
    
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()