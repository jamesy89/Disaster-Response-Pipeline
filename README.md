# Disaster Response Pipeline Project

This is a project to classify text to different categories based on the words in a sentence in order to direct them to the appropriate response unit. The project involves reading and merging 2 datlasets, sanitizing the data and saving it to a SQLite database to be used for training by a ML pipeline.

A function to tokenize, normalize, and lemmatize the text was then implemented, followed by training a Multi-output classifier and fine tuning the parameters using grid search. Finally, a web app is used to visualize the results of the model.

The `\data\process_data.py` file includes the python implementation to load and prepare the dataset and then saving it to a SQLite database for future use.

The `\models\train_classifier.py` file contains the python implementation to load the data from a SQLite database and trains a ML model that will classify text into the 36 different categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
