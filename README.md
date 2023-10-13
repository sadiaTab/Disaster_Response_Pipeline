# Disaster Response Pipeline 
![App header](https://raw.githubusercontent.com/sadiaTab/Disaster_Response_Pipeline/main/screenshots/header.png?token=GHSAT0AAAAAACIODOYRZ6XQXFGNBZBVI2ZAZJJHMGQ)

## Project Overview
The Udacity Disaster Response Pipeline project is a web application that analys## Project Components

The project consists of the following components:

- ETL (Extract, Transform, Load) Pipeline: Processes and cleans the data.
- ML (Machine Learning) Pipeline: Trains and tests a multi-output classification model.
- Flask Web App: Provides a user interface for classifying new messages.

## ETL Pipeline
The ETL pipeline loads and processes the data from various sources, performs data cleaning, and stores the clean data in a SQLite database. This data is later used for training the machine learning model.

## ML Pipeline
The ML pipeline includes data preprocessing, model training, and model evaluation. It uses a multi-output classification model to categorize messages into different disaster-related categories.

## Flask Web App
The Flask web app offers a user-friendly interface to input a message, which is then classified into relevant categories using the trained model. Users can also explore visualizations of the dataset.

## Running
To run the project, follow these steps:

### Data Cleaning
1. Run the ETL pipeline to clean and store the data:
   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```

## How to run the app

Run following commands:
- python process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- python train_classifier.py data/DisasterResponse.db models/classifier.pkl
- python run.py