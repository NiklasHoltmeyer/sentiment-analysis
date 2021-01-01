from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf

from DeepSentiment.Consts import (
    Global as Global, 
    Glove as Glove, 
    Paths as Paths, 
    Preprocessing as Preprocessing, 
    Training as Training 
)

from os import path
import sys
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

class Callbacks:
    def __init__(self, args):
        self.args = args
        self.earlyStopping = EarlyStopping(
                        monitor='val_loss',  #loss
                        #monitor='accuracy',
                        min_delta=args['early_stopping_min'],
                        patience=args['early_stopping_patience'],
                        restore_best_weights=True)
        
        self.reduceLRonPlateau = ReduceLROnPlateau(monitor='val_loss', patience=args['early_stopping_patience'], cooldown=0)
    
        self.modelCheckpoint = lambda checkpoint_path : ModelCheckpoint(filepath=checkpoint_path + "/cp-{epoch:04d}.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=args["train_batch_size"] * 5) #every poch # batch_size*5 = every 5th epoch)

        self.csvLogger = lambda filePath : CSVLogger(filePath, separator=';', append=True)
    
    def createCheckpointPath(self, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):        
        modelFolderName = Logging(self.args).createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        path = Path(Paths.MODEL_CHECKPOINTS, modelFolderName)
        path.mkdir(parents=True, exist_ok=True)
        
        return str(path.resolve())
    
    def createModelPath(self, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):        
        modelFolderName = Logging(self.args).createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        path = Path(Paths.MODEL, modelFolderName)
        path.mkdir(parents=True, exist_ok=True)
        filePath = Path(path, "{}.tf".format(modelFolderName))
        return str(filePath.resolve())
    
class Logging:
    def __init__(self, args):
        self.args = args
        
    def createModelName(self, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):
        keyValues = {   "GLOVE" : GLOVE, 
                        "CNN" : CNN_LAYER, 
                        "POOLING" : POOLING_LAYER, 
                        "GRU" : GRU_LAYER, 
                        "BiLSTM" : BiLSTM_Layer,
                        "LSTM" : LSTM_Layer, 
                        "DENSE" : DENSE_LAYER}
        
        modelName = "_".join([key for key, value in keyValues.items() if value])
        modelName = modelName if len(modelName) > 0 else "no_name"
        return modelName

    def createLogPath(self, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER, PREFIX = "", SUFFIX=".log"):        
        modelName = self.createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        pathFolder = Path(Paths.MODEL_CHECKPOINTS, modelName)
        pathFolder.mkdir(parents=True, exist_ok=True)
        pathFile = Path(pathFolder, "{}{}{}".format(PREFIX, modelName, SUFFIX))
        return str(pathFile.resolve())

    def getLogger(self, loggingFile = None, consoleLogging = True, logginLevel = logging.DEBUG, loggingPrefix = ""):
        logFileHandler = logging.FileHandler(filename=loggingFile) if loggingFile is not None else None
        consoleHandler = logging.StreamHandler(sys.stdout) if consoleLogging else None
        
        handlers = [logFileHandler, consoleHandler]
        handlers = list(filter(None, handlers))
        
        logger=DeepLogger.defaultLogger(handlers=handlers)       
    
        return logger
    
    def getResultCSVRow(self, history, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):        
        name = self.createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)        
        
        acc = history.history['accuracy']
        vallAcc = history.history['val_accuracy']
        
        epochs = len(history.history['val_accuracy'])
        accMean, accMax = np.mean(acc), max(acc)
        vallAccMean, vallAccMax = np.mean(vallAcc), max(vallAcc)
        
        layersCSV = ";".join([("x" if x else "") for x in [GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER]])
        
        time = str(datetime.now()).replace(" ", "_")
        dataset = self.args["number_of_training_data_entries"]
        row = ";".join([str(x) for x in [time, dataset, epochs, name,  layersCSV, accMean, accMax, vallAccMean, vallAccMax ]])
        return row + "\n"
    
    def loggingResult(self, history, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):
        row  = self.getResultCSVRow(history=history, GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        path, name = Paths.TRAINING_RESULT_CSV
        filePath = Path(path, name).resolve()
        Path(path).mkdir(parents=True, exist_ok=True)
                
        with open(filePath, "a+") as f:
            f.write(row)
        
        
class Encoder:
    def trainWordVectorEncoder(trainText, 
                               VOCAB_SIZE=None):
        #https://www.tensorflow.org/tutorials/text/text_classification_rnn
        encoder = TextVectorization() if VOCAB_SIZE is None else  TextVectorization(max_tokens=VOCAB_SIZE)    
        encoder.adapt(tf.data.Dataset.from_tensor_slices(trainText))
    
        return encoder
