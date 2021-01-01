# region Imports
import logging
from DeepSentiment.Dataset import (Glove as GloveDS, Sentiment140 as Sentiment140DS)
from DeepSentiment.Networks.Tensorflow.Model import Model as TFModel

from DeepSentiment.Consts import (
    #Global as Global, 
    #Glove as Glove, 
    Paths as Paths, 
    #Preprocessing as Preprocessing, 
    Training as Training,
    SimpleTransformers as SimpleTransformersConsts
)

from DeepSentiment.Networks.Tensorflow.Helper import(Callbacks, Encoder, Logging)

from DeepSentiment.Preprocessing.CleanText import CleanText
from DeepSentiment.Logging import Logger as DeepLogger

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
from pprint import pformat
import torch

from pathlib import Path
# endregion

class Model:
    def __init__(self, model_type, model_name, logger=DeepLogger.defaultLogger()):
        self.model_type = model_type
        self.model_name = model_name
        self.logger = logger     
        self.model = None  
        
    def train(self, args={}, cleanFN = CleanText().cleanText, size=Training.NUMBER_OF_TRAINING_DATA_ENTRIES):
        self.logger.debug("Train Simpletransformer")
        isCudaAvailable = torch.cuda.is_available()
        
        if not isCudaAvailable:
            self.logger.warning("Training on CPU!")
        
        _modelArgs = self.modelArgs(args)           
        
        self.logger.debug("ModelArgs: ")     
        self.logger.debug("\n" + pformat(_modelArgs))
        trainData, testData = self.loadDataset(cleanFN=cleanFN, size=size)
        
        self.model = ClassificationModel(model_type=self.model_type, model_name=self.model_name, args=_modelArgs, 
                            use_cuda=isCudaAvailable, 
                            num_labels=2)
        
        self.model.train_model(train_df=trainData, eval_df=testData)        
       
        return self.model
    
    def loadDataset(self, cleanFN, size):    
        train_data, test_data, labelEncoder, s140 = TFModel().loadDataset(LOAD_GLOVE = False, 
                                                                           padInput = False, 
                                                                           logger = self.logger,
                                                                           Tokanize = False,
                                                                          BERT = True)
        
        test_data  = test_data.rename(columns={"sentiment": "labels"})
        train_data = train_data.rename(columns={"sentiment": "labels"})

        test_data["labels"] = test_data["labels"].apply(lambda x: np.int8(1) if x in 'Positive' else np.int8(0))
        train_data["labels"] = train_data["labels"].apply(lambda x: np.int8(1) if x in 'Positive' else np.int8(0))
        
        sizeSecond = int(size * 0.2) if size is not None else None        
        
        train_shuffeld = train_data.sample(n=size, random_state=42) if size is not None else train_data.sample(n=len(train_data), random_state=42)
        test_shuffeld = test_data.sample(n=sizeSecond, random_state=42)  if sizeSecond is not None else test_data.sample(n=len(test_data), random_state=42)
        
        return train_shuffeld, test_shuffeld
    
    def load(self, folder, modelName="model"): 
        ''' Path = e.G. f"{Paths.RESULTS_BASE}/transformer_{model_name}_e5_{size}" '''
        absolutePath = str(Path(folder, modelName).resolve())
        self.model = torch.load(absolutePath)
        return self.model
        
    def save(self, folder, modelName="model"): 
        ''' Path = e.G. f"{Paths.RESULTS_BASE}/transformer_{model_name}_e5_{size}" '''
        if self.model is not None:
            self.logger.debug(f"Saving Model ({path})")
            
            _path = Path(folder)
            _path.mkdir(parents=True, exist_ok=True)
            
            absolutePath = str(Path(folder, modelName).resolve())
            
            torch.save(self.model, absolutePath)
            self.logger.debug("Saved Model")
            
        else:
            self.logger.error("Coul not Save Model!")
        
    
    def modelArgs(self, args={}):
        _modelArgs = SimpleTransformersConsts.MODEL_ARGS
        
        for k, v in args.items(): 
            _modelArgs[k] = v
        
        return _modelArgs