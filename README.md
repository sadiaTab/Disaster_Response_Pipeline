# Disaster Response Pipeline 
![App header]("https://github.com/sadiaTab/Disaster_Response_Pipeline/blob/main/screenshots/1_sc.png")
In this project thousands of real messages sent during natural disasters either through social media or directly to disaster response organisations.
Build an ETL pipeline that processes message and category data from the csv files and load them to a SQLite database.
Build a Machine learning pipeline which will read the database, create and save a multi output Supervised learning model.
A web app will extract the data from the database to provide data visualisations, and use  ML model to classify new messages for 36 categories.
Machine Learning is important to help different organisations to understand which messages are relevant to them and which messages to priotarise.
During a disaster, they have the least capacity to filter out messages that matters. 

## How to run the app

Run following commands:
- python process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- python train_classifier.py data/DisasterResponse.db models/classifier.pkl
- python run.py