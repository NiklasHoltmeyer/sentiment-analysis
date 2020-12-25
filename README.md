# Deep Learning for Sentiment Analysis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NiklasHoltmeyer/sentiment-analysis/blob/main/example/Deep%20Learning%20for%20Sentiment%20Analysis%20-%20example.ipynb)

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/NiklasHoltmeyer/sentiment-analysis) [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 [![Github](https://img.shields.io/badge/Git-Hub-green.svg)](https://github.com/NiklasHoltmeyer/sentiment-analysis)

Sentiment Analysis trained on Sentiment140

## Installation
The following script contains the (Python) requirements and training data.


```bash
bash ./scripts/0_install_prerequisites.sh
```

## Usage
### Training
```python
import consts as CONSTS # GLOBAL, PATHS, GLOVE, TRAINING, PREPROCESSING
from TensorflowHelper import Callbacks, Encoder, Logging
from TensorflowModels import TensorflowModels
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format= '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', #%(asctime)s - %(levelname)s: %(message)s
)

logger = logging.getLogger("sentiment")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("nltk_data").setLevel(logging.WARNING)

model, history = TensorflowModels().trainModel(CNN_LAYER = True,  #self-trained word2vec embedding layer
            POOLING_LAYER = True, 
            BiLSTM_Layer = True, 
            logger = logger)
            
#model, history = TensorflowModels().trainModel(GLOVE = True, 
#            CNN_LAYER = True 
#            POOLING_LAYER = True 
#            GRU_LAYER = True 
#            BiLSTM_Layer = True 
#            LSTM_Layer = True 
#            DENSE_LAYER = True,
#            logger = logger)

```
### Prediction
#### self-trained embedding layer (Word2Vec)
```python
from clean_text import CleanText
sample_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
sample_text_cleaned = CleanText().cleanText(sample_text)
model.predict([sample_text])
```

#### GloVe embedding layer (subject to change)
```python
from Sentiment140Dataset import Sentiment140Dataset
from clean_text import CleanText

s140 = Sentiment140Dataset(path=CONSTS.PATHS.SENTIMENT140_DATASET, 
                                parsedPath=CONSTS.PATHS.SENTIMENT140_DATASET_PARSED,
                                embeddingDim=CONSTS.GLOVE.GLOVE_DIM, 
                                MAX_SEQUENCE_LENGTH=CONSTS.PREPROCESSING.MAX_SEQUENCE_LENGTH, 
                                logger = logger)
train_data, test_data, labelDecoder = s140.load(padInput=True, 
                                                TRAIN_SIZE=CONSTS.TRAINING.TRAIN_SIZE, 
                                                        DEBUG=CONSTS.GLOBAL.DEBUG, 
                                                        cleanFN = CleanText().cleanText,
                                                        Tokanize = True,
                                                        BERT = False)

sample_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
sample_text_cleaned = CleanText().cleanText(sample_text)
sample_text_glove = s140.padInput(sample_text_cleaned)
model.predict([sample_text_glove])
```
