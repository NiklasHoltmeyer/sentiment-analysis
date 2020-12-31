import numpy as np
import logging
import time

class Dataset:            
    def __init__(self, path, dimension, logger):
        self.path = path
        self.dimension = dimension
        self.logger = logger
        self.read()        
    
    def read(self):
        embeddings_index = {}
        
        self.logger.debug('Reading Glove Vectors [{} Dimensions]'.format(self.dimension))
        startTime = time.time()

        with open(self.path, 'r', encoding="utf8") as gloveFile:
            #Based on: https://www.kaggle.com/arunrk7/nlp-beginner-text-classification-using-lstm
            for line in gloveFile:
                tokens = line.split()
                word = tokens[0]
                vector = np.array(tokens[1:], dtype=np.float32)

                embeddings_index[word] = vector
        self.embeddings_index = embeddings_index
        
        elapsedTime =  time.time() - startTime
        self.logger.debug('Readin Glove Vectors [DONE] - {} seconds'.format(elapsedTime))
        
        return embeddings_index

    def embeddingMatrix(self, tokenizer):
        #https://www.kaggle.com/arunrk7/nlp-beginner-text-classification-using-lstm
        vocab_size = len(tokenizer.word_index) + 1
        #dim = self.dim
        embedding_matrix = np.zeros((vocab_size, self.dimension))

        for word, i in tokenizer.word_index.items():
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix