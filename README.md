# Disaster-Response-Pipelines
Udacity DSND Project 3

## Installation <a name="installation"></a>
This code runs on Python  

### Libraries
to make this project work you need to import Sklearn, NLTK, Plotly, Sqlalchemy, Pickle, and Flask

## Instructions

1. run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the following command to run your web app.
    `python run.py`
4. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

this project aims to create a ML pipeline for disaster response using data from Figure8 and displaying it 
on a web app. 

## File Descriptions <a name="files"></a>

1. app folder 
  - run.py
  - templates folder
    -- master.html
    -- go.html
2. data folder
  - disaster_categories.csv
  - disaster_messages.csv
  - DisasterResponse.db
  - process_data.py
3. models folder
  - train_classifier.py
  - classifier.pkl
4. screenshots folder
  - classification report.png
  - classification 1.1.png
  - classification 1.2.png
  - classification 2.1.png
  - classification 2.1.png
  - web app 1.png
  - web app 2.png
  - web app 3.png
  - web app 4.png

## Results<a name="results"></a>
the ML model scored an accuracy of 0.943973006186965 which could be improved, a screenshot of the classification report is shown in screenshots folder in addition to screenshots of the web app.

## Acknowledgements<a name="licensing"></a>
I would like to acknowledge Udacity for providing this project's templates and Figure8 for the data. 


