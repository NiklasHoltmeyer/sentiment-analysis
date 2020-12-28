from sklearn.preprocessing import LabelEncoder
#from consts import PATHS, GLOVE
import consts as CONSTS
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import logging
import time
import os.path
import csv
import sys
import numpy as np 
class Sentiment140Dataset:        
    def __init__(self, path, parsedPath, embeddingDim, logger, MAX_SEQUENCE_LENGTH=CONSTS.PREPROCESSING.MAX_SEQUENCE_LENGTH):
        self.path = path
        self.parsedPath = parsedPath
        self.tokenizer = None
        self.embeddingDim = embeddingDim
        self.MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH
        self.logger = logger
        
    decodeSentiment = lambda x: "Positive" if x == 4 else "Negative" if x == 0 else "ERROR"
        
    def load(self, cleanFN, TRAIN_SIZE=0.8, padInput=True, DEBUG=False, Tokanize=True, BERT=False):
        # Read Data
        self.logger.debug('[Sentiment140] Reading Sentiment Dataset')       
        startTime = time.time()  
        dataset = self.__load_dataset(cleanFN)
                    
        data_rows = CONSTS.TRAINING.NUMBER_OF_TRAINING_DATA_ENTRIES if CONSTS.TRAINING.NUMBER_OF_TRAINING_DATA_ENTRIES is not None else len(dataset)
                
        dataset = dataset.sample(n=data_rows, random_state=42)
        
        self.logger.debug('[Sentiment140] Reading Sentiment Dataset [DONE] - {} seconds'.format(time.time() - startTime))
        self.logger.debug('[Sentiment140] Clean Sentiment Dataset')
        startTime = time.time()        
        
        self.logger.debug('[Sentiment140] Clean Sentiment Dataset (Sentiment)')
        dataset['sentiment'] = dataset['sentiment'].apply(Sentiment140Dataset.decodeSentiment)
        
        if DEBUG:
            dataset['sentiment'].hist()

        #Tokenizer
        self.logger.debug('[Sentiment140] Clean Sentiment Dataset [Done]')   
        train_data, test_data = train_test_split(dataset, test_size=1-TRAIN_SIZE, random_state=7)    
       
        self.logger.debug('[Sentiment140] Tokanize Text')   
        if Tokanize:
            self.tokenizer = self.loadTokenizer(train_data.text)
                
        #Padding Input
        x_train, x_test = train_data.text, test_data.text
        
        if padInput:
            self.logger.debug('[Sentiment140] Pad Input')   
            x_train, x_test = self.padInput(train_data.text), self.padInput(test_data.text)
        else:
            self.logger.debug('[Sentiment140] Pad Input (disabled)') 
            
        if BERT:
            return train_data, test_data, None
        
        #Transform Output
        self.logger.debug('[Sentiment140] LabelEncode')
        labelEncoder, y_train, y_test = self.transformLabel(targets = train_data.sentiment, trainingCorpus = train_data.sentiment, validationCorpus = test_data.sentiment)
        
        self.logger.debug('[Sentiment140] LabelEncode [DONE]')
                
        return ((x_train, y_train), (x_test, y_test), labelEncoder) 
    
    def __load_dataset(self, cleanFN, forceReload = False, depth=0):
        '''
            Load Sentiment140 from Disk 
            ForceReload = force to reparse Data 
        '''
        isDatasetParsed = os.path.isfile(CONSTS.PATHS.SENTIMENT140_DATASET_PARSED) and not forceReload
        
        filePath = self.parsedPath if isDatasetParsed else self.path
        
        dataset = pd.read_csv(filePath, encoding ="ISO-8859-1" , names=["sentiment", "ids", "date", "flag", "user", "text"])                                      
        dataset['text'].replace('', np.nan, inplace=True)
        dataset.dropna(subset=['text'], inplace=True)
           
        if not isDatasetParsed:
            startTime = time.time()                    
            self.logger.debug('[Sentiment140] Clean Sentiment Dataset (Text)')
            dataset['text'] = dataset['text'].apply(cleanFN) 
            self.logger.debug('[Sentiment140] Clean Sentiment Dataset [DONE] - {} seconds'.format(time.time() - startTime))
            dataset.to_csv(self.parsedPath, encoding ="ISO-8859-1", index=False, header=False, quoting=csv.QUOTE_ALL)           
            
            if depth > 1:
                self.logger.error('[Sentiment140] Reading Data cyclic recursion')
                sys.exit(1)
            
            return self.__load_dataset(cleanFN, forceReload, depth=depth+1)
        else:
            self.logger.debug('[Sentiment140] Clean Sentiment Dataset [Done] (Read from Drive)')   
        return dataset.drop(["ids", "date", "flag", "user"], axis=1)
    
    # , names=["sentiment", "ids", "date", "flag", "user", "text"])
        
            
    
    def transformLabel(self, targets, trainingCorpus, validationCorpus):
        labelEncoder = LabelEncoder()
        labelEncoder.fit(targets)

        y_train = labelEncoder.transform(trainingCorpus).reshape(-1, 1)
        y_test   = labelEncoder.transform(validationCorpus).reshape(-1, 1)
        
        return (labelEncoder, y_train, y_test)
    
    def loadTokenizer(self, corpus):
        tokenizer = Tokenizer()  
        tokenizer.fit_on_texts(corpus)
        self.vocab_size  = len(tokenizer.word_index) + 1
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index        
        return tokenizer
    
    def getTokenizer(self):
        return self.tokenizer
    
    def padInput(self, text): #should be inside glove
        if self.tokenizer is None:
            print("ERROR: first setTokenizer!")
            return None
        return pad_sequences(self.tokenizer.texts_to_sequences(text),
                        maxlen = self.MAX_SEQUENCE_LENGTH)  

    ##def getEmbeddingLayer(self, embeddingMatrix, embeddingDim):          
        ####direkt returnen lässt sich nicht so gut in das model integrieren - deshalb über params func gehen
        ##embeddingLayer = tf.keras.layers.Embedding(self.vocab_size,
                                          ##self.embeddingDim,
                                          ##weights=[embeddingMatrix],
                                          ##input_length=self.MAX_SEQUENCE_LENGTH,
                                          ##trainable=False)
        ##return (embeddingLayer, MAX_SEQUENCE_LENGTH)
    
    def getEmbeddingLayerParams(self, embeddingMatrix, embeddingDim):               
        return (self.vocab_size, self.embeddingDim, [embeddingMatrix], self.MAX_SEQUENCE_LENGTH)


