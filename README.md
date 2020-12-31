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
from DeepSentiment.Networks.Tensorflow.Model import Model as TFModel
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format= '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', #%(asctime)s - %(levelname)s: %(message)s
)

logger = logging.getLogger("sentiment")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("nltk_data").setLevel(logging.WARNING)

model, history = TFModel().trainModel(CNN_LAYER = True,  #self-trained word2vec embedding layer
            POOLING_LAYER = True, 
            BiLSTM_Layer = True, 
            logger = logger)
            
#model, history = TFModel().trainModel(GLOVE = True, 
#            CNN_LAYER = True 
#            POOLING_LAYER = True 
#            GRU_LAYER = True 
#            BiLSTM_Layer = True 
#            LSTM_Layer = True 
#            DENSE_LAYER = True,
#            logger = logger)

```
### Load Model
```python
model = TFModel().loadModel(CNN_LAYER = True,  #self-trained word2vec embedding layer
            POOLING_LAYER = True, 
            BiLSTM_Layer = True, 
            logger = logger)
```
### Prediction
#### self-trained embedding layer (Word2Vec)
```python
from DeepSentiment.Preprocessing.CleanText import CleanText

sample_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
sample_text_cleaned = CleanText().cleanText(sample_text)
model.predict([sample_text])
```

#### GloVe embedding layer (subject to change)
```python
from DeepSentiment.Preprocessing.CleanText import CleanText
from DeepSentiment.Dataset import Sentiment140
from DeepSentiment.Consts import (
    Global as Global, 
    Glove as Glove, 
    Paths as Paths, 
    Preprocessing as Preprocessing, 
    Training as Training 
)


s140 = Sentiment140.Dataset(path=Paths.SENTIMENT140_DATASET, 
                                parsedPath=Paths.SENTIMENT140_DATASET_PARSED,
                                embeddingDim=Glove.GLOVE_DIM, 
                                MAX_SEQUENCE_LENGTH=Preprocessing.MAX_SEQUENCE_LENGTH, 
                                logger = logger)
train_data, test_data, labelDecoder = s140.load(padInput=True, 
                                                TRAIN_SIZE=Training.TRAIN_SIZE, 
                                                        DEBUG=Global.DEBUG, 
                                                        cleanFN = CleanText().cleanText,
                                                        Tokanize = True,
                                                        BERT = False)

sample_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
sample_text_cleaned = CleanText().cleanText(sample_text)
sample_text_glove = s140.padInput(sample_text_cleaned)
predictions = model.predict([sample_text_glove])
```
