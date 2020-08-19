import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle
from sklearn.ensemble import RandomForestClassifier
from functools import lru_cache

def load_data(database_filepath):
    ''' 
    loads data from an SQL datbase

    INPUT: database file path
    OUTPUT: predictor and target variables as well as category names 

    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    dbname = database_filepath.split('/')[-1]
    df = pd.read_sql_table(dbname.split('.')[0], con = engine)
    
    X = df['message']
    Y = df.iloc[:,4:]

    #240 columns in related attribute have a value of 2, so they are mapped to 1
    for ind in Y.index: 
        if(Y['related'][ind] == 2):
            Y['related'][ind] = 1

    category_names = Y.columns

    return X,Y,category_names

stop_set = set(stopwords.words())
lemmatizer = WordNetLemmatizer()

@lru_cache(maxsize=10000)
def lemmatize(w):
    ''' 
    lemmatizes a text word

    INPUT: word
    OUTPUT: lemmetized word

    '''
    return lemmatizer.lemmatize(w.lower().strip())

def tokenize(text):
    ''' 
    tokenizes text

    INPUT: text
    OUTPUT: list of words from the text in their root form

    '''

    #normalize text
    text=text.lower()
    text=re.sub(r"[^a-zA-Z0-9]"," ", text)

    #tokenize words
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stop_set]

    # Reduce words to their root form
    lemmed = [lemmatize(w) for w in words]

    return lemmed

def build_model():
    ''' 
    builds ML model for message classification

    INPUT: none
    OUTPUT: ML model that performs best 

    '''
    #pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # model parameters to preform GridSearchCV on
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [5, 10],
                }

    #model 
    model = GridSearchCV(estimator=pipeline, 
            param_grid=parameters, cv=2, verbose=2)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    evaluates ML model for message classification by printing 
    classification report and accuracy score of chosen model

    INPUT: model, X_test, Y_test, category_names
    OUTPUT: none

    '''
    # prediction
    y_pred = model.predict(X_test)

    # classification report 
    print(classification_report(Y_test.values, y_pred, target_names=category_names, zero_division=0))

    # model accuracy 
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    ''' 
    saves model into a pickle file

    INPUT: model and its file path
    OUTPUT: none
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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