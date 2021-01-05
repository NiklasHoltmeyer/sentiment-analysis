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
        
    def train(self, args={}, cleanFN = CleanText().cleanText):
        self.logger.debug("Train Simpletransformer")
        isCudaAvailable = torch.cuda.is_available()
        
        if not isCudaAvailable:
            self.logger.warning("Training on CPU!")
        
        _modelArgs = self.modelArgs(args)             
        
        self.logger.debug("ModelArgs: ")     
        self.logger.debug("\n" + pformat(_modelArgs))
        self.loadData(cleanFN, _modelArgs)
        
        self.model = ClassificationModel(model_type=self.model_type, model_name=self.model_name, args=_modelArgs, 
                            use_cuda=isCudaAvailable, 
                            num_labels=2)
        
        return self.model.train_model(train_df=self.trainData, eval_df=self.testData)        
    
    def loadData(self, cleanFN, args={}):
        _modelArgs = self.modelArgs(args)   
        self.trainData, self.testData = self.loadDataset(cleanFN=cleanFN, args=_modelArgs)  #Dataframe if "lazy_loading" False, else TSV Path
           
    def validate(self, data, args={}):
        #_modelArgs = self.modelArgs(args)
        self.logger.debug("Validate Simpletransformer Modell")
        return self.model.eval_model(data) #result, model_outputs, wrong_predictions 
    
    def loadDataset(self, cleanFN, args={}):    
        _modelArgs = self.modelArgs(args)
        train_data, test_data, labelEncoder, s140 = TFModel().loadDataset(LOAD_GLOVE = False, 
                                                                           padInput = False, 
                                                                           Tokanize = False,
                                                                          BERT = True, args=_modelArgs)
        
        test_data  = test_data.rename(columns={"sentiment": "labels"})
        train_data = train_data.rename(columns={"sentiment": "labels"})

        test_data["labels"] = test_data["labels"].apply(lambda x: np.int8(1) if x in 'Positive' else np.int8(0))
        train_data["labels"] = train_data["labels"].apply(lambda x: np.int8(1) if x in 'Positive' else np.int8(0))
        
        if "lazy_loading" in _modelArgsd:
            trainingPath, testPath = Paths.SENTIMENT140_DATASET_PARSED_TSV
            
            return train_data.to_csv(trainingPath, sep="\t"), test_data.to_csv(testPath, sep="\t")            

#        size = _modelArgs["number_of_training_data_entries"]
#        sizeSecond = int(size * 0.2) if size is not None else None        
        
#        train_shuffeld = train_data.sample(n=size, random_state=42) if size is not None else train_data.sample(n=len(train_data), random_state=42)
#        test_shuffeld = test_data.sample(n=sizeSecond, random_state=42)  if sizeSecond is not None else test_data.sample(n=len(test_data), random_state=42)
        
        return train_data, test_data
    
    def load(self, folder, modelName="model"): 
        ''' Path = e.G. f"{Paths.RESULTS_BASE}/transformer_{model_name}_e5_{size}" '''
        absolutePath = str(Path(folder, modelName).resolve())
        if not torch.cuda.is_available():
            self.logger.warning("Training/Predicting on CPU!")        
            self.model = torch.load(absolutePath, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(absolutePath)
        return self.model
        
    def save(self, folder, modelName="model"): 
        ''' Path = e.G. f"{Paths.RESULTS_BASE}/transformer_{model_name}_e5_{size}" '''
        if self.model is not None:
            _path = Path(folder)
            _path.mkdir(parents=True, exist_ok=True)
            absolutePath = str(Path(folder, modelName).resolve())
            
            self.logger.debug(f"Saving Model ({absolutePath})")
            
            torch.save(self.model, absolutePath)
            self.logger.debug("Saved Model")
            
        else:
            self.logger.error("Coul not Save Model!")
        
    
    def mapKey(self, key):
        if key in Training.mapKeys:
            return Training.mapKeys[key]
        return key
    
    def modelArgs(self, args={}):
        _modelArgs = SimpleTransformersConsts.MODEL_ARGS
        
        for k, v in args.items(): 
            key = self.mapKey(k)
            _modelArgs[key] = v
        
        return _modelArgs