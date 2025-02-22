import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    ''' 
    loads messages and categories datasets from csv file 
    and merges them

    INPUT: csv file paths
    OUTPUT: dataframe of marged datasets
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    ''' 
    cleans the raw data of the categories

    INPUT: dataframe
    OUTPUT: cleaned dataframe
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0] 

    # extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    categories.head() 

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

     #240 columns in related attribute have a value of 2, so they are mapped to 1
    for ind in df.index: 
        if(df['related'][ind] == 2):
            df['related'][ind] = 1

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_filename):
    ''' 
    saves dataframe into an SQL database

    INPUT: dataframe and database file name
    OUTPUT: none
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    dbname = database_filename.split('/')[-1]
    df.to_sql(dbname.split('.')[0], engine, index=False, if_exists='replace') 

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