# -*- coding: utf-8 -*-

from setuptools import setup #, find_packages

with open('LICENSE') as f:
    license = f.read()
    
#with open('README.rst') as f:
    #readme = f.read()
    
setup(
    name='DeepSentiment',
    version='0.1.0',
    description='Deep Learning for Sentiment Analysis',
    #long_description=readme,
    author='Niklas Holtmeyer',
    url='https://github.com/NiklasHoltmeyer/sentiment-analysis',
    license=license,
    packages=['DeepSentiment', 'DeepSentiment.Consts', 'DeepSentiment.Dataset', 'DeepSentiment.Networks', 'DeepSentiment.Preprocessing', 'DeepSentiment.Networks.Transformer'],#find_packages(exclude=('tests', 'docs')),
    install_requires=[        
        'tensorflow==2.4.0', 
        'pandas', 
        'matplotlib', 
        'nltk', 
        'chakin', 
        'contractions', 
        'emoji', 
        'transformers', 
        'simpletransformers', 
        'tqdm>=4.47.0'         
    ]
        #'tensorflow==2.4.0',
        #'pandas', 
        #'numpy', 
        #'matplotlib==3.3.3', 
        #'nltk==3.5', 
        #'chakin==0.0.8', 
        #'contractions==0.0.43', 
        #'emoji==0.6.0', 
        #'transformers==4.1.1', 
        #'simpletransformers==0.51.5', 
        #'tqdm', 
)