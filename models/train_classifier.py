import sys
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.pipeline import Pipeline
nltk.download('wordnet')
nltk.download('punkt')


def load_data(database_filepath):
    """Loads the sqlite database from the given filepath and returns the features, targets, and target names"""
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    #df.head()
    X = df['message']
    Y = df.iloc[:,4:]
    categories = list(Y)

    return X, Y, categories


def tokenize(text):
    """Tokenize, normalize, and lemmatize a given string"""
    # Normalize
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Lemmatize
    lmtzr = WordNetLemmatizer()
    lemmatized = [lmtzr.lemmatize(word) for word in tokens]

    return lemmatized


def build_model():
    """Creates a ML pipeline for training the model, and initializes the GridSearchCV for tuning model"""
    # Create ML pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(MultinomialNB())),])

    # Parameters to grid search
    parameters = parameters = {'tfidf__use_idf': (True, False),
                           'vect__ngram_range': [(1, 1), (1, 2)],
                           'tfidf__smooth_idf': (True, False),}

    # Grid search
    grid_obj = GridSearchCV(pipeline, parameters, n_jobs=-1)

    return grid_obj

def evaluate_model(model, X_test, Y_test, category_names):
    """Uses the best estimator found by grid search to get predictions on the test set and print the results"""
    # Get best estimator found by grid search
    best_clf = model.best_estimator_
    # Get predictions on test set
    best_predictions = best_clf.predict(X_test)

    # Print results
    print(classification_report(Y_test, best_predictions, target_names=category_names))


def save_model(model, model_filepath):
    """Saves the trained model to a pickle file at the given path"""
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model = model.fit(X_train, Y_train)

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
