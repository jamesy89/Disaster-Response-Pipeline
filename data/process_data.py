import sys
import numpy as np
import pandas as pd
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv("disaster_messages.csv")
    #messages.head()

    # load categories dataset
    categories = pd.read_csv("disaster_categories.csv")
    #categories.head()

    # merge datasets
    df = pd.merge(messages, categories, on='id')
    #df.head()

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';', expand=True)
    #categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[[0]]
    #print(row)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = []
    for column in row:
        category = row[column][0]
        category_name = category[:-2]
        category_colnames.append(category_name)

        #print(category_colnames)
        # rename the columns of `categories`
        categories.columns = category_colnames
        #categories.head()

        for column in categories:
        #print(categories[column])
        # set each value to be the last character of the string
        value = categories[column].str[-1:]
        categories[column] = value

        #print(categories[column][1])
        # convert column from string to numeric
        categories[column] = int(categories[column][1])
    #categories.head()

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #df.head()

    # check number of duplicates
    duplicated = df.duplicated()
    #duplicated.head()

    num_duplicate = 0
    for duplicate in duplicated:
        if duplicate == True:
            num_duplicate += 1

    #print(num_duplicate)

    # drop duplicates
    df = df.drop_duplicates(keep=False)

    # replace NaN with 0
    df = df.fillna(0)

    return df


def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('InsertTableName', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
