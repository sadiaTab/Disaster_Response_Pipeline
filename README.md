# Disaster Response Pipeline Project
![App header](https://github.com/sadiaTab/Disaster_Response_Pipeline/blob/main/screenshots/sc_header.png?raw=true)

## 1. Project Overview
The Udacity Disaster Response Pipeline project is a web application that analyzes and classifies messages related to disasters. The goal of this project is to help emergency responders quickly identify the most relevant messages during a disaster, enabling faster response and assistance.

## 2. Project Components

The project consists of the following components:

### 2.1. ETL Pipeline
- Loads the data from `messages` and `categories` datasets
- Processes and cleans the data
- Stores the clean data in a **SQLite database**
- ETL pipeline code can be found in **_data/process_data.py_**

### 2.2. Machine Learning Pipeline
- Loads data from the **SQLite database**
- Trains a multi-output classification model. 
- Saves the final model as a pickle file
- The model is used to categorise messages into different disaster-related categories.
- Machine Learning pipeline code can be found in **_models/train_classifier.py_**


### 2.3. Web Application
- The Flask web app offers a user-friendly interface to input a message, which is then classified into relevant categories using the trained model.
- Users can also explore visualizations of the dataset.



## 3. Getting Started

### 3.1. Dependencies
- **Python Version**: Python 3.11 
- **Machine Learning Libraries**: 
  - NumPy
  - SciPy
  - Pandas
  - Scikit-Learn
- **Natural Language Processing Libraries**: 
  - NLTK (Natural Language Toolkit)
- **SQLite Database Libraries**: 
  - SQLAlchemy
- **Model Loading and Saving Library**: 
  - Pickle
- **Web Application and Data Visualization Libraries**: 
  - Flask
  - Plotly

### 3.2. Cloning the repository

To clone the git repository: 

    
        git clone https://github.com/sadiaTab/Disaster_Response_Pipeline.git
    

To run the project, follow these steps:

### 3.3. Running the project

1. Run the ETL pipeline to clean and store the processed data in the database:
   
   ```bash
   python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
    ```
   ![Loading Data](https://github.com/sadiaTab/Disaster_Response_Pipeline/blob/main/screenshots/loading_data.png?raw=true)
   
3. Run the Machine Learning pipeline, which involves loading data from a database, training a classifier, and saving the classifier as a pickle file:
   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```
    ![Saving Data](https://github.com/sadiaTab/Disaster_Response_Pipeline/blob/main/screenshots/save_model.png?raw=true)
   
4. Run the following command in the app directory:
 ```bash
   python run.py
   ```
4. Go to `http://127.0.0.1:3000/` to use the web application.

## 4. Screenshots

- This is the app frontpage showing training data distribution and word cloud.

![App front](https://github.com/sadiaTab/Disaster_Response_Pipeline/blob/main/screenshots/training_data_distribution.png?raw=true)

- This is the result page showing the classification of new message into different disaster-related categories by the machine learning model.

![App front](https://github.com/sadiaTab/Disaster_Response_Pipeline/blob/main/screenshots/result.png?raw=true)

## 5. Acknowledgement

This app was created as part of the ![Udacity Data Scientist Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The original dataset used in this project was initially curated by Udacity in partnership with ![Figure Eight (Appen)](https://appen.com/).