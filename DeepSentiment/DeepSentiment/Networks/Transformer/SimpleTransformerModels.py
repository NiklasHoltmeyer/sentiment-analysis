import consts as CONSTS # GLOBAL, PATHS, GLOVE, TRAINING, PREPROCESSING
from TensorflowHelper import Callbacks, Encoder, Logging
from TensorflowModels import TensorflowModels
import logging
from DeepSentiment.Dataset import (Glove as GloveDS, Sentiment140 as Sentiment140DS)

from DeepSentiment.Consts import Training as Training 

class SimpleTransformerModels:
    def __init__(model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name
    
    def loadDataset(self, LOAD_GLOVE=False, padInput=False, logger, cleanFN = CleanText().cleanText, Tokanize=False, BERT = True):    
        train_data, test_data, labelEncoder, s140 = TensorflowModels().loadDataset(LOAD_GLOVE = LOAD_GLOVE, 
                                                                           padInput = padInput, 
                                                                           logger = logger,
                                                                           Tokanize = Tokanize,
                                                                          BERT = BERT)
        
        test_data = test_data.rename(columns={"sentiment": "labels"})
        train_data =train_data.rename(columns={"sentiment": "labels"})

        test_data["labels"] = test_data["labels"].apply(lambda x: 1 if x in 'Positive' else 0)
        train_data["labels"] = train_data["labels"].apply(lambda x: 1 if x in 'Positive' else 0)
        
        size = Training.NUMBER_OF_TRAINING_DATA_ENTRIES
        sizeSecond = int(size * 0.2) if size is not None else None        
        
        train_shuffeld = train_data.sample(n=size, random_state=42) if size is not None else train_data.sample(n=len(train_data), random_state=42)
        test_shuffeld = test_data.sample(n=sizeSecond, random_state=42)  if sizeSecond is not None else test_data.sample(n=len(test_data), random_state=42)
        
        return train_shuffeld, test_shuffeld