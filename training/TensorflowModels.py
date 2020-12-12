
#region Imports
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
import os

import re
import string

from pathlib import Path

import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, Embedding, AveragePooling1D, MaxPooling1D, Flatten, GRU, MaxPool1D, SpatialDropout1D
from tensorflow.keras import Sequential

#Glove
import chakin # download glove
from collections import defaultdict
import logging
import sys
##
from clean_text import CleanText
import consts as CONSTS # GLOBAL, PATHS, GLOVE, TRAINING, PREPROCESSING
from GloveDataset import GloveDataset 
from Sentiment140Dataset import Sentiment140Dataset
from TensorflowHelper import Callbacks, Encoder, Logging

from tensorflow.keras.optimizers import Adam

class TensorflowModels:
    def loadDataset(self, LOAD_GLOVE, padInput, logger):    
        s140 = Sentiment140Dataset(path=CONSTS.PATHS.SENTIMENT140_DATASET, 
                                embeddingDim=CONSTS.GLOVE.GLOVE_DIM, 
                                MAX_SEQUENCE_LENGTH=CONSTS.PREPROCESSING.MAX_SEQUENCE_LENGTH, 
                                logger = logger)

        train_data, test_data, labelDecoder = s140.load(TRAIN_SIZE=CONSTS.TRAINING.TRAIN_SIZE, 
                                                        DEBUG=CONSTS.GLOBAL.DEBUG, 
                                                        cleanFN = CleanText().cleanText,
                                                        padInput=padInput)

        gloveEmbeddingMatrix = GloveDataset(CONSTS.GLOVE.GLOVE_FILENAME, CONSTS.GLOVE.GLOVE_DIM, logger = logger) \
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

    def addGRULayer(self, model):
        model.add(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        return model

    def addLSTMLayer(self, model):
        model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1))
        return model

    def addBiLSTMLayer(self, model):
        model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1)))
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

    def baseModelGlove(self, logger):
        #load Sequential Model & Glove-Embedding-Layer, with Training / Test Dataset
        train_data, test_data, labelDecoder, s140, gloveEmbeddingMatrix = self.loadDataset(LOAD_GLOVE=True, padInput=True, logger=logger) 
        vocab_size, embeddingDim, embeddingMatrix, MAX_SEQUENCE_LENGTH = s140.getEmbeddingLayerParams(gloveEmbeddingMatrix, 
                                                                                                    CONSTS.GLOVE.GLOVE_DIM)
        model = self.embeddingLayerGloveModel(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, 
                                        vocab_size=vocab_size, 
                                        embeddingDim=embeddingDim, 
                                        embeddingMatrix=embeddingMatrix)    
        return model, train_data, test_data

    def baseModelNonGlove(self, logger):
        #load Sequential Model & Embedding-Layer, with Training / Test Dataset

        train_data, test_data, labelDecoder, s140 = self.loadDataset(LOAD_GLOVE=False, padInput=False, logger=logger)
        trainX, trainY = train_data
        encoder = Encoder.trainWordVectorEncoder(trainX, VOCAB_SIZE=CONSTS.PREPROCESSING.MAX_SEQUENCE_LENGTH)
        model = self.embeddingLayerNoGloveModel(encoder) 
        return model, train_data, test_data

    def createModel(self, GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False, logger = None):
        model, train_data, test_data = self.baseModelGlove(logger) if GLOVE else self.baseModelNonGlove(logger)
        trainX, trainY = train_data
        logger.debug(("[Model] with Glove" if GLOVE else "[Model] Selftrained Word2Vec"))
        
        if CNN_LAYER:
            logger.debug("[Model] Add CNN_LAYER")
            model = self.addCNNLayer(model)
        
        if POOLING_LAYER:
            logger.debug("[Model] Add POOLING_LAYER")
            model = self.addCNNLayer(model)
            
        if GRU_LAYER:
            logger.debug("[Model] Add GRU_LAYER")
            model = self.addGRULayer(model)
        
        if BiLSTM_Layer:        
            logger.debug("[Model] Add BiLSTM_Layer")
            model = self.addBiLSTMLayer(model)

        if LSTM_Layer:
            logger.debug("[Model] Add LSTM_Layer")
            model = self.addLSTMLayer(model)
        
        if DENSE_LAYER:
            logger.debug("[Model] Add DENSE_LAYER")
            model = self.addDenseLayer(model)
        
        logger.debug("Dataset: Training = {}, Validation = {} Item(s)".format(len(train_data[0]), len(test_data[0])))
        model.add(Dense(1, activation='sigmoid')) #output layer
        
        model.compile(optimizer=Adam(learning_rate=CONSTS.TRAINING.Learning_Rate), loss='binary_crossentropy', metrics=['accuracy'])    
        
        return model, trainX, trainY, test_data
    
    def loadModel(self, GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False, logger = None):
        model, _, _, _ = self.createModel(GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, LSTM_Layer, BiLSTM_Layer, DENSE_LAYER, logger)
        modelPath = Callbacks.createModelPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        tf.keras.models.load_model(modelPath) #load_weights load_model
        return model
    
    def trainModel(self, GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False, logger = None):
        model, trainX, trainY, test_data = self.createModel(GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, LSTM_Layer, BiLSTM_Layer, DENSE_LAYER, logger)
            
        checkPointPath = Callbacks.createCheckpointPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        csvLoggerPath = Logging.createLogPath(PREFIX = "keras_", SUFFIX=".csv", GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        modelPath = Callbacks.createModelPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        
        callsBacks = [Callbacks.earlyStopping, Callbacks.reduceLRonPlateau, Callbacks.modelCheckpoint(checkPointPath), Callbacks.csvLogger(csvLoggerPath)]
            
        history = model.fit(trainX, trainY, epochs=CONSTS.TRAINING.EPOCHS, 
                            batch_size=CONSTS.TRAINING.BATCH_SIZE,
                            validation_data=test_data, 
                            callbacks=callsBacks,
                            verbose=2)
                            
        model.summary()
        model.save(modelPath)          
            
        return model, history
            