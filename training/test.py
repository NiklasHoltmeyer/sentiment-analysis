import consts as CONSTS
import pandas as pd
import numpy as np
from Sentiment140Dataset import Sentiment140Dataset
import logging
###
from clean_text import CleanText

import os.path

logging.basicConfig(
    level=logging.DEBUG, 
    format= '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', #%(asctime)s - %(levelname)s: %(message)s
)

logger = logging.getLogger("sentiment")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("nltk_data").setLevel(logging.WARNING)

s140 = Sentiment140Dataset(path=CONSTS.PATHS.SENTIMENT140_DATASET, 
                        parsedPath = CONSTS.PATHS.SENTIMENT140_DATASET_PARSED,
                        embeddingDim=CONSTS.GLOVE.GLOVE_DIM, 
                        MAX_SEQUENCE_LENGTH=CONSTS.PREPROCESSING.MAX_SEQUENCE_LENGTH, 
                        logger = logger)
cleanFN = CleanText().cleanText
train_data, test_data, labelDecoder = s140.load(TRAIN_SIZE=CONSTS.TRAINING.TRAIN_SIZE, 
                                                DEBUG=CONSTS.GLOBAL.DEBUG, 
                                                cleanFN = cleanFN,
                                                padInput=True,
                                                Tokanize = True,
                                                BERT = False)
