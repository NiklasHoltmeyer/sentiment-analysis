#!/bin/bash 
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

mkdir -p "$SCRIPTPATH/../data/sentiment140"
wget -nc https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/training.1600000.processed.noemoticon.csv.zip -P ../data/sentiment140
unzip -n -d "$SCRIPTPATH/../data/sentiment140/" "$SCRIPTPATH/../data/sentiment140/training.1600000.processed.noemoticon.csv.zip"

mkdir -p "$SCRIPTPATH/../data/glove"
wget -nc http://nlp.stanford.edu/data/glove.twitter.27B.zip -P "$SCRIPTPATH/../data/glove"
unzip -n -d "$SCRIPTPATH/../data/glove" "$SCRIPTPATH/../data/glove"/glove.twitter.27B.zip

python -m pip install -r requirements.txt
