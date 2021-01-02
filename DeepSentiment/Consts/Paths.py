import os

BASE = os.getcwd()
RESULTS_BASE = os.getcwd() + "/results/"
MODEL_CHECKPOINTS = RESULTS_BASE + "/model_checkpoints/"
MODEL = RESULTS_BASE + "/model/"
SENTIMENT140_DATASET = "I:\workspace\sentiment-analysis\data\sentiment140/training.1600000.processed.noemoticon.csv" ##BASE +"\\Datasets\Sentiment140_twitter\\training.1600000.processed.noemoticon.csv"
SENTIMENT140_DATASET_PARSED = "I:\workspace\sentiment-analysis\data\sentiment140/training_parsed.csv"
SENTIMENT140_URL = "https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/training.1600000.processed.noemoticon.csv.zip"
    
IMDB_DATASET = "I:\workspace\sentiment-analysis\data\aclImdb/aclImdb.csv"
IMDB_DATASET_Parsed = "I:\workspace\sentiment-analysis\data\aclImdb/aclImdb_parsed.csv"
TRAINING_RESULT_CSV = (RESULTS_BASE, "result.csv") #path, filename