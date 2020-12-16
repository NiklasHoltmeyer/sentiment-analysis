import contractions
import re
import string

import nltk
from nltk.corpus import stopwords  
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

class CleanText:     
    _firstInstance = True #<- Stemmer-Daten runterladen, falls erste Instanz
    stemmerFN = None       
    lemmatizeFn = None
    stopWords = None
    cleanPattern = None
    
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
    
    def __init__(self):    
        if CleanText._firstInstance:
            CleanText._firstInstance = False
            
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet') #<- für lemmatazing
            
            CleanText.stopWords = stopwords.words('english') #the, ...
            CleanText.stemmerFN = SnowballStemmer("english").stem #zum finden des wort-stammes   
            CleanText.cleanPattern = CleanText.regpexpCleanPatterns()
            
            CleanText.lemmatizeFn = WordNetLemmatizer().lemmatize
            
    def cleanText(self, text):
        '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        #Based on: https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model 
        # https://www.kaggle.com/avramandrei96/short-and-simple-lstm-with-glove-embeddings
        # ...
        text = str(text).lower()
        text = self.replaceEmojiis(text)
        
        for pattern, replace in CleanText.cleanPattern:
            text = re.sub(pattern, replace, text)
        
        text = self.removestopWordsAndNonAlphabetic(text)               
        text = self.lemmantizing(text)
        text = contractions.fix(text).lower() ##bspw. you've -> you have
        #output_data = tf.strings.regex_replace(output_data,"(\s){2,}", "")   #multiple whitespaces
        
        return text
      
    def regpexpCleanPatterns():
        #Based on: https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model 
        # https://www.kaggle.com/avramandrei96/short-and-simple-lstm-with-glove-embeddings
        return [
                #('(:|;)(\)|d|>)', 'happy'), # :), :D, :>, ;), ;D, ;>
                #('(\(|c|<)(:|;)', 'happy'), # (:, c:, <:, (;, c;, <;
                #('(:|;)(\(|c|<)', 'sad'), #  :( :c :<,  ;( ;c ;<, 
                #('(\)|>)(:|;)', 'sad'), # ); >;
                ('\[.*?\]', ''), 
                ('https?://\S+|www\.\S+', ''), 
                ('<.*?>+', ''), 
                ('[%s]' % re.escape(string.punctuation), ''), 
                ('\n', ''), 
                ('\w*\d\w*', ''),     
                ('<[^>]+>', ''),  #html
                ('@[a-zA-Z0-9_]+', 'user'), ##@...      
                ('#[a-zA-Z0-9_]+', 'hashtag'), #tag
                ('%', "prozent"), #noise
                ("\x89", ""), #noise
                ("hÛ_", ""), #noise
                ("ÛÓ", "") #noise
        ]
    
    def replaceEmojiis(CleanText, text):
        #SRC: https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners 
        for emoji in CleanText.emojis.keys():
            text = text.replace(emoji, CleanText.emojis[emoji])  
        return text        
    
    def lemmantizing(self, text):
        tokens = []
        for token in text.split():
            if token not in CleanText.stopWords:
                tokens.append(CleanText.lemmatizeFn(token))
        return " ".join(tokens)   
    
    def stemming(self, text):
        tokens = []
        for token in text.split():
            if token not in CleanText.stopWords:
                tokens.append(CleanText.stemmerFN(token))
        return " ".join(tokens)    
    
    def removestopWordsAndNonAlphabetic(self,text):
        #based On https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/
        tokens = nltk.word_tokenize(text)
        return " ".join([token for token in tokens if token.isalpha() and token not in CleanText.stopWords])
