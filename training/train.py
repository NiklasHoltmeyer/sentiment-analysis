#region Imports
import argparse

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
#endregion

#path ="S:\OneDrive - Hochschule Osnabrück\Semester 3\Fachseminar\Hausarbeit\Code\\"
#sys.path.insert(0, path)

#region logging

#endregion

def str2bool(v):
    #src: https://stackoverflow.com/a/43357954/5026265
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def loadDataset(LOAD_GLOVE, padInput, logger):    
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

def addCNNLayer(model):    
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=64, kernel_size=5, padding = 'same', activation='relu'))    
    return model

def addPoolingLayer(model):
    model.add(MaxPool1D(pool_size = 2))
    return model

def addGRULayer(model):
    model.add(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    return model

def addLSTMLayer(model):
    model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1))
    return model

def addBiLSTMLayer(model):
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=1)))
    return model

def addDenseLayer(model):
    model.add(Dense(128, activation='relu'))
    return model

def embeddingLayerGloveModel(MAX_SEQUENCE_LENGTH, vocab_size, embeddingDim, embeddingMatrix):
    model = tf.keras.Sequential()
        
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model.add(tf.keras.layers.Embedding(vocab_size,
                                            embeddingDim,
                                            weights=embeddingMatrix,
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False))              
    
    return model

def embeddingLayerNoGloveModel(encoder):
    model = tf.keras.Sequential()
    
    model.add(encoder) 
    model.add(tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
            output_dim=128,
            # Use masking to handle the variable sequence lengths
            mask_zero=True))  
    return model

def baseModelGlove(logger):
    #load Sequential Model & Glove-Embedding-Layer, with Training / Test Dataset
    train_data, test_data, labelDecoder, s140, gloveEmbeddingMatrix = loadDataset(LOAD_GLOVE=True, padInput=True, logger=logger) 
    vocab_size, embeddingDim, embeddingMatrix, MAX_SEQUENCE_LENGTH = s140.getEmbeddingLayerParams(gloveEmbeddingMatrix, 
                                                                                                  CONSTS.GLOVE.GLOVE_DIM)
    model = embeddingLayerGloveModel(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, 
                                     vocab_size=vocab_size, 
                                     embeddingDim=embeddingDim, 
                                     embeddingMatrix=embeddingMatrix)    
    return model, train_data, test_data

def baseModelNonGlove(logger):
    #load Sequential Model & Embedding-Layer, with Training / Test Dataset

    train_data, test_data, labelDecoder, s140 = loadDataset(LOAD_GLOVE=False, padInput=False, logger=logger)
    trainX, trainY = train_data
    encoder = Encoder.trainWordVectorEncoder(trainX, VOCAB_SIZE=CONSTS.PREPROCESSING.MAX_SEQUENCE_LENGTH)
    model = embeddingLayerNoGloveModel(encoder) 
    return model, train_data, test_data

def testModel(GLOVE = False, CNN_LAYER = False, POOLING_LAYER = False, GRU_LAYER = False, LSTM_Layer = False, BiLSTM_Layer = False, DENSE_LAYER = False, logger = None):
    model, train_data, test_data = baseModelGlove(logger) if GLOVE else baseModelNonGlove(logger)
    trainX, trainY = train_data
    logger.debug(("[Model] with Glove" if GLOVE else "[Model] Selftrained Word2Vec"))
    
    if CNN_LAYER:
        logger.debug("[Model] Add CNN_LAYER")
        model = addCNNLayer(model)
    
    if POOLING_LAYER:
        logger.debug("[Model] Add POOLING_LAYER")
        model = addCNNLayer(model)
        
    if GRU_LAYER:
        logger.debug("[Model] Add GRU_LAYER")
        model = addGRULayer(model)
    
    if BiLSTM_Layer:        
        logger.debug("[Model] Add BiLSTM_Layer")
        model = addBiLSTMLayer(model)

    if LSTM_Layer:
        logger.debug("[Model] Add LSTM_Layer")
        model = addLSTMLayer(model)
    
    if DENSE_LAYER:
        logger.debug("[Model] Add DENSE_LAYER")
        model = addDenseLayer(model)
    
    logger.debug("Dataset: Training = {}, Validation = {} Item(s)".format(len(train_data[0]), len(test_data[0])))
    model.add(Dense(1, activation='sigmoid')) #output layer
    
    model.compile(optimizer=Adam(learning_rate=CONSTS.TRAINING.Learning_Rate), loss='binary_crossentropy',
                metrics=['accuracy'])    

    checkPointPath = Callbacks.createCheckpointPath(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, LSTM_Layer = LSTM_Layer, BiLSTM_Layer = BiLSTM_Layer, DENSE_LAYER = DENSE_LAYER)
    
    history = model.fit(trainX, trainY, epochs=CONSTS.TRAINING.EPOCHS, 
                        batch_size=CONSTS.TRAINING.BATCH_SIZE,
                        validation_data=test_data, 
                        callbacks=[Callbacks.earlyStopping, Callbacks.reduceLRonPlateau, Callbacks.modelCheckpoint(checkPointPath)],
                        verbose=2)
    
    model.summary()
    
    return model, history
  
parser = argparse.ArgumentParser()
parser.add_argument('--glove', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Should the Model use Glove or should it train its own WordVec")

parser.add_argument('--layer_CNN',         type=str2bool, nargs='?', const=True, default=False, help="Add a CNN Layer")
parser.add_argument('--layer_POOLING',     type=str2bool, nargs='?', const=True, default=False, help="Add a POOLING Layer")
parser.add_argument('--layer_GRU',         type=str2bool, nargs='?', const=True, default=False, help="Add a GRU Layer")
parser.add_argument('--layer_BiLSTM',      type=str2bool, nargs='?', const=True, default=False, help="Add a BiLSTM Layer")
parser.add_argument('--layer_LSTM',        type=str2bool, nargs='?', const=True, default=False, help="Add a LSTM Layer")
parser.add_argument('--layer_DENSE',       type=str2bool, nargs='?', const=True, default=False, help="Add a DENSE Layer")

args = parser.parse_args()

loggingFile = Logging.createLogPath(GLOVE = args.glove, 
            CNN_LAYER = args.layer_CNN, 
            POOLING_LAYER = args.layer_POOLING, 
            GRU_LAYER = args.layer_GRU, 
            BiLSTM_Layer = args.layer_BiLSTM, 
            LSTM_Layer = args.layer_LSTM, 
            DENSE_LAYER = args.layer_DENSE)

logger = Logging.getLogger(loggingFile = loggingFile, consoleLogging = True, logginLevel = logging.DEBUG)

model, history = testModel(GLOVE = args.glove, 
            CNN_LAYER = args.layer_CNN, 
            POOLING_LAYER = args.layer_POOLING, 
            GRU_LAYER = args.layer_GRU, 
            BiLSTM_Layer = args.layer_BiLSTM, 
            LSTM_Layer = args.layer_LSTM, 
            DENSE_LAYER = args.layer_DENSE,
            logger = logger)

Logging.loggingResult(history, GLOVE = args.glove, 
            CNN_LAYER = args.layer_CNN, 
            POOLING_LAYER = args.layer_POOLING, 
            GRU_LAYER = args.layer_GRU, 
            LSTM_Layer = args.layer_LSTM, 
            BiLSTM_Layer = args.layer_BiLSTM, 
            DENSE_LAYER = args.layer_DENSE)
#Bsp:
#python train.py --glove 1 
#  --layer_CNN 1 \
#  --layer_POOLING 1 \
#  --layer_GRU 1 \
#  --layer_LSTM 0 \
#  --layer_DENSE 1 \
#  >> glove_cnn_pooling_gru_dense.txt
