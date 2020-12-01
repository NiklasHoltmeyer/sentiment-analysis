import os

class GLOBAL:
    DEBUG = True
    #os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

class PATHS: 
    BASE = os.getcwd()
    SENTIMENT140_DATASET = "data/sentiment140/training.1600000.processed.noemoticon.csv" ##BASE +"\\Datasets\Sentiment140_twitter\\training.1600000.processed.noemoticon.csv"
    TRAINING_RESULT_CSV = (os.getcwd(), "result.csv") #path, filename
    
class GLOVE:
    GLOVE_FILENAME = "data/glove/glove.twitter.27B.100d.txt"
    GLOVE_DIM = 100
    
class TRAINING:
    TRAIN_SIZE = 0.8
    EPOCHS = 10
    Learning_Rate = 1e-3
    EARLY_STOPPING_MIN = 1e-3
    EARLY_STOPPING_PATIENCE = 5
    NUMBER_OF_TRAINING_DATA_ENTRIES = None# <- None = read every entry -> faster testing with less data
    BATCH_SIZE = 1024
    
class PREPROCESSING:
    MAX_SEQUENCE_LENGTH = 30#144 #1000
    
    


