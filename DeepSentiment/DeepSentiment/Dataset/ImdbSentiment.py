import os
import shutil 
import tensorflow as tf
from random import shuffle
from glob import glob
from itertools import chain
import csv
from pathlib import Path
import pandas as pd 
import pathlib
import numpy as np

class Dataset:  
    cacheFiles = []
    fileEncodig = 'UTF-8'
    columns = ['text', 'sentiment', 'ds', 'fileName']
    
    def __init__(self, path, parsedPath):
        self.path = path
        self.parsedPath = parsedPath
        
    def load(self, cleanFN, forceReload = False):        
        isDatasetParsed = os.path.isfile(self.parsedPath) and not forceReload
        isDataSetAvailable = os.path.isfile(self.path)
        
        if not isDataSetAvailable:
            self.download_imdb()
            self.convertFilesToCSV(cleanFN)
            
        dataset = pd.read_csv(self.parsedPath, encoding=self.fileEncodig , names=self.columns, quoting=csv.QUOTE_ALL)                                      
        dataset['text'].replace('', np.nan, inplace=True)
        dataset.dropna(subset=['text'], inplace=True)
        
        return dataset
        
        
    
    def download_imdb(self):
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

        dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

        train_dir = os.path.join(dataset_dir, 'train')

        # remove unused folders to make it easier to load the data
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)
        
        self.cacheFiles.append("aclImdb_v1.tar.gz")
        self.cacheFiles.append("aclImdb")   
        
        parentPath = pathlib.Path(self.path).parent.absolute()
        
        Path(parentPath).mkdir(parents=True, exist_ok=True)
        
        return "aclImdb"
        
        
    def readText(self, path):
        with open(path, encoding=self.fileEncodig) as f:
            return f.read().replace('"', "'")
        return None
    
    
    def readSentiment(self, filePath):
        fileName = os.path.basename(filePath).replace(".txt", "")
        sentiment = "pos" if "pos" in filePath else "neg"
        ds = "train" if "train" in filePath else "test"
        text = self.readText(filePath)
        return (text, sentiment, ds, fileName)
    
    def cleanSentiment(self, sentiment):
        return 1 if "pos" in sentiment else 0
    
    def convertFilesToCSV(self, cleanFN):
        filePaths = list(chain.from_iterable(glob(f'aclImdb/{ds}/{sent}/*.txt')
                                 for ds in ('test', 'train') for sent in ('pos', 'neg')))
        
        data = [self.readSentiment(path) for path in filePaths]
        df = pd.DataFrame.from_records(data, columns =self.columns) 
        
        df.to_csv(self.path, encoding=self.fileEncodig, index=False, header=False, quoting=csv.QUOTE_ALL)   
        df['text'] = df['text'].apply(cleanFN) 
        df['sentiment'] = df['sentiment'].apply(self.cleanSentiment) 
        
        df.to_csv(self.parsedPath, encoding=self.fileEncodig, index=False, header=False, quoting=csv.QUOTE_ALL)   
        
    
    def removeCache(self):
        for file in self.cacheFiles:
            os.remove(file)