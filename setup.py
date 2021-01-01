# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

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
    scripts=["scripts/install_prerequisites.sh"],    
    packages=find_packages(exclude=('tests', 'docs')),
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    install_requires=[
        "tqdm",#==4.41.1
        'pandas==1.1.5', #1.2.0
        'matplotlib==3.3.3',
        'chakin==0.0.8',
        'contractions==0.0.43',
        'emoji==0.6.0',
        'nltk==3.5',
        'scikit-learn==0.23.2',
        'scipy==1.5.4',
        'tensorboard==2.4.0',
        'tensorboard-plugin-wit==1.7.0',
        'tensorboardx==2.1',
        'tensorflow',
        'tensorflow-estimator',
        'tensorflow-metadata==0.26.0',
        'tfa-nightly==0.13.0.dev20201223200403',
        'tfds-nightly==4.1.0.dev202012260107',
        'tokenizers==0.9.4',
        'torch==1.6.0+cu101',
        'torchvision==0.7.0+cu101',
        #'torch',
        #'torchvision',
        'transformers',
        'simpletransformers',
    ],
)
