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
    #Zumteil - src: Tensorflow Tutorial, https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis
    earlyStopping = EarlyStopping(
        monitor='val_loss',  #loss
        #monitor='accuracy',
        min_delta=Training.EARLY_STOPPING_MIN, # minimium amount of change to count as an improvement
        patience=Training.EARLY_STOPPING_PATIENCE, # anzahl epochen die gewartet werden sollen
        restore_best_weights=True)

    reduceLRonPlateau = ReduceLROnPlateau(monitor='val_loss', patience=Training.EARLY_STOPPING_PATIENCE, cooldown=0)
    
    modelCheckpoint = lambda checkpoint_path : ModelCheckpoint(filepath=checkpoint_path + "/cp-{epoch:04d}.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=Training.BATCH_SIZE * 5) #every poch # batch_size*5 = every 5th epoch)

    csvLogger = lambda filePath : CSVLogger(filePath, separator=';', append=True)
    
    def createCheckpointPath(GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):        
        modelFolderName = Logging.createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        path = Path(Paths.MODEL_CHECKPOINTS, modelFolderName)
        path.mkdir(parents=True, exist_ok=True)
        
        return str(path.resolve())
    
    def createModelPath(GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):        
        modelFolderName = Logging.createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        path = Path(Paths.MODEL, modelFolderName)
        path.mkdir(parents=True, exist_ok=True)
        filePath = Path(path, "{}.tf".format(modelFolderName))
        return str(filePath.resolve())
    
class Logging:
    def createModelName(GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):
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

    def createLogPath(GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER, PREFIX = "", SUFFIX=".log"):        
        modelName = Logging.createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
        pathFolder = Path(Paths.MODEL_CHECKPOINTS, modelName)
        pathFolder.mkdir(parents=True, exist_ok=True)
        pathFile = Path(pathFolder, "{}{}{}".format(PREFIX, modelName, SUFFIX))
        return str(pathFile.resolve())

    def getLogger(loggingFile = None, consoleLogging = True, logginLevel = logging.DEBUG, loggingPrefix = ""):
        logFileHandler = logging.FileHandler(filename=loggingFile) if loggingFile is not None else None
        consoleHandler = logging.StreamHandler(sys.stdout) if consoleLogging else None
        
        handlers = [logFileHandler, consoleHandler]
        handlers = list(filter(None, handlers))
        
        logging.basicConfig(
            level=logginLevel, 
            format= loggingPrefix + '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', #%(asctime)s - %(levelname)s: %(message)s
            handlers=handlers
        )
        
        logger = logging.getLogger("sentiment")
        
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("nltk_data").setLevel(logging.WARNING)
        return logger
    
    def getResultCSVRow(history, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):        
        name = Logging.createModelName(GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)        
        
        acc = history.history['accuracy']
        vallAcc = history.history['val_accuracy']
        
        epochs = len(history.history['val_accuracy'])
        accMean, accMax = np.mean(acc), max(acc)
        vallAccMean, vallAccMax = np.mean(vallAcc), max(vallAcc)
        
        layersCSV = ";".join([("x" if x else "") for x in [GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER]])
        
        time = str(datetime.now()).replace(" ", "_")
        dataset = Training.NUMBER_OF_TRAINING_DATA_ENTRIES
        row = ";".join([str(x) for x in [time, dataset, epochs, name,  layersCSV, accMean, accMax, vallAccMean, vallAccMax ]])
        return row + "\n"
    
    def loggingResult(history, GLOVE, CNN_LAYER, POOLING_LAYER, GRU_LAYER, BiLSTM_Layer, LSTM_Layer, DENSE_LAYER):
        row  = Logging.getResultCSVRow(history=history, GLOVE = GLOVE, CNN_LAYER = CNN_LAYER, POOLING_LAYER = POOLING_LAYER, GRU_LAYER = GRU_LAYER, BiLSTM_Layer = BiLSTM_Layer, LSTM_Layer = LSTM_Layer, DENSE_LAYER = DENSE_LAYER)
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
