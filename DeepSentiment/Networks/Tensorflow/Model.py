# region Imports
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
import os

import re
import string

from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, Embedding, AveragePooling1D, MaxPooling1D, Flatten, GRU, MaxPool1D, SpatialDropout1D
from tensorflow.keras import Sequential

#Glove
from collections import defaultdict
import logging
import sys
##

from DeepSentiment.Dataset import (Glove as GloveDS, Sentiment140 as Sentiment140DS)

from tensorflow.keras.optimizers import Adam

from DeepSentiment.Logging import Logger as DeepLogger
from DeepSentiment.Consts import (
    Global as Global, 
    Glove as Glove, 
    Paths as Paths, 
    Preprocessing as Preprocessing, 
    Training as Training 
)

from DeepSentiment.Networks.Tensorflow.Helper import(
    Callbacks, Encoder, Logging
)

from DeepSentiment.Preprocessing.CleanText import CleanText
from pprint import pformat
# endregion

class Model:
    def __init__(self, logger=DeepLogger.defaultLogger()):
        self.logger = logger     
        
    def loadDataset(self, LOAD_GLOVE, padInput, args, cleanFN = CleanText().cleanText, Tokanize=True, BERT = False):    
        s140 = Sentiment140DS.Dataset(path=Paths.SENTIMENT140_DATASET, 
                                   parsedPath = Paths.SENTIMENT140_DATASET_PARSED,
                                embeddingDim=Glove.GLOVE_DIM, 
                                MAX_SEQUENCE_LENGTH=Preprocessing.MAX_SEQUENCE_LENGTH, 
                                logger = self.logger, args=args)

        train_data, test_data, labelDecoder = s140.load(DEBUG=Global.DEBUG, 
                                                        cleanFN = cleanFN,
                                                        padInput=padInput,
                                                        Tokanize = Tokanize,
                                                        BERT = BERT)

        gloveEmbeddingMatrix = GloveDS.Dataset(Glove.GLOVE_FILENAME, Glove.GLOVE_DIM, logger = self.logger) \
                                .embeddingMatrix(s140.getTokenizer()) if LOAD_GLOVE else None

        return (train_data, test_data, labelDecoder, s140, gloveEmbeddingMatrix) if LOAD_GLOVE else \
            (train_data, test_data, labelDecoder, s140)

    
    def addCNNLayer(self, model):    
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=64, kernel_size=5, padding = 'same', activation='relu'))    
        return model

    def addPoolingLayer(self, model):
        model.add(MaxPool1D(pool_size = 2))
        return model

    def addGRULayer(self, model, return_sequences=False):
        model.add(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1))
        model.add(GRU(256, return_sequences=return_sequences))
        return model

    def addLSTMLayer(self, model, return_sequences=False):
        model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1))
        model.add(LSTM(256, return_sequences=return_sequences))
        return model

    def addBiLSTMLayer(self, model, return_sequences=False):
        model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1)))
        model.add(Bidirectional(LSTM(256, return_sequences=return_sequences)))
        return model

    def addDenseLayer(self, model):
        model.add(Dense(128, activation='relu'))
        return model    
    
    def embeddingLayerGloveModel(self, MAX_SEQUENCE_LENGTH, vocab_size, embeddingDim, embeddingMatrix):
        model = tf.keras.Sequential()
            
        model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
        model.add(tf.keras.layers.Embedding(vocab_size,
                                                embeddingDim,
                                                weights=embeddingMatrix,
                                                input_length=MAX_SEQUENCE_LENGTH,
                                                trainable=False))              
        
        return model

    def embeddingLayerNoGloveModel(self, encoder):
        model = tf.keras.Sequential()
        
        model.add(encoder) 
        model.add(tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                output_dim=128,
                # Use masking to handle the variable sequence lengths
                mask_zero=True))  
        return model

    def baseModelGlove(self, args):
        #load Sequential Model & Glove-Embedding-Layer, with Training / Test Dataset
        train_data, test_data, labelDecoder, s140, gloveEmbeddingMatrix = self.loadDataset(LOAD_GLOVE=True, padInput=True, args=args) 
        vocab_size, embeddingDim, embeddingMatrix, MAX_SEQUENCE_LENGTH = s140.getEmbeddingLayerParams(gloveEmbeddingMatrix, 
                                                                                                    Glove.GLOVE_DIM)
        model = self.embeddingLayerGloveModel(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, 
                                        vocab_size=vocab_size, 
                                        embeddingDim=embeddingDim, 
                                        embeddingMatrix=embeddingMatrix)    
        return model, train_data, test_data

    def baseModelNonGlove(self, args):
        #load Sequential Model & Embedding-Layer, with Training / Test Dataset

        train_data, test_data, labelDecoder, s140 = self.loadDataset(LOAD_GLOVE=False, padInput=False, args=args)
        trainX, trainY = train_data
        encoder = Encoder.trainWordVectorEncoder(trainX, VOCAB_SIZE=Preprocessing.MAX_SEQUENCE_LENGTH)
        model = self.embeddingLayerNoGloveModel(encoder) 
        return model, train_data, test_data

    
    def createModel(self, args, GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False):
        model, train_data, test_data = self.baseModelGlove(args=args) if GLOVE else self.baseModelNonGlove(args=args)
        trainX, trainY = train_data
        self.logger.debug(("[Model] with Glove" if GLOVE else "[Model] Selftrained Word2Vec"))

        if CNN_LAYER:
            self.logger.debug("[Model] Add CNN_LAYER")
            model = self.addCNNLayer(model)
        
        if POOLING_LAYER:
            self.logger.debug("[Model] Add POOLING_LAYER")
            model = self.addPoolingLayer(model)  
        
        if GRU_LAYER:
            self.logger.debug("[Model] Add GRU_LAYER")
            returnSequences = BiLSTM_Layer or LSTM_Layer # only if next layer = RNN
            model = self.addGRULayer(model, returnSequences)
        
        if BiLSTM_Layer:        
            self.logger.debug("[Model] Add BiLSTM_Layer")
            returnSequences = LSTM_Layer # only if next layer = RNN
            model = self.addBiLSTMLayer(model, returnSequences)

        if LSTM_Layer:
            self.logger.debug("[Model] Add LSTM_Layer")
            returnSequences = False
            model = self.addLSTMLayer(model, returnSequences)
        
        if DENSE_LAYER:
            self.logger.debug("[Model] Add DENSE_LAYER")
            model = self.addDenseLayer(model)
        
        self.logger.debug("Dataset: Training = {}, Validation = {} Item(s)".format(len(train_data[0]), len(test_data[0])))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid')) #output layer      
        
        model.compile(optimizer=Adam(learning_rate=args['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])    
        
        return model, trainX, trainY, test_data
    
    def loadModel(self, GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False, args={}):
        _args = self.modelArgs(args)
        modelPath = Callbacks(_args).createModelPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        return tf.keras.models.load_model(modelPath) 
    
    def trainModel(self, GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False, args={}):
        _args = self.modelArgs(args)
        
        self.logger.debug("ModelArgs: ")     
        self.logger.debug("\n" + pformat(_args))
        
        model, trainX, trainY, test_data = self.createModel(args = _args, GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        
        callBackHelper = Callbacks(_args)
        checkPointPath = callBackHelper.createCheckpointPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        csvLoggerPath = Logging(_args).createLogPath(PREFIX = "keras_", SUFFIX=".csv", GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        modelPath = callBackHelper.createModelPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        
        callsBacks = [callBackHelper.earlyStopping, callBackHelper.reduceLRonPlateau, callBackHelper.modelCheckpoint(checkPointPath), callBackHelper.csvLogger(csvLoggerPath)]
            
        history = model.fit(trainX, trainY, epochs=_args['num_train_epochs'], 
                            batch_size=_args['train_batch_size'],
                            validation_data=test_data, 
                            callbacks=callsBacks,
                            verbose=2)
                            
        model.summary()
        model.save(modelPath)          
            
        return model, history
    
    def mapKey(self, key):
        if key in Training.mapKeys:
            return Training.mapKeys[key]
        return key

    def modelArgs(self, args={}):
        _modelArgs = Training.trainArgs        
       
        for k, v in args.items(): 
            key = self.mapKey(k)
            _modelArgs[key] = v
        
        return _modelArgs
            
