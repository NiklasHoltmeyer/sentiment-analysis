#!/bin/bash 
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# Sentiment
mkdir -p "$SCRIPTPATH/../data/sentiment140"
wget -nc https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/training.1600000.processed.noemoticon.csv.zip -P "$SCRIPTPATH/../data/sentiment140"
unzip -n -d "$SCRIPTPATH/../data/sentiment140/" "$SCRIPTPATH/../data/sentiment140/training.1600000.processed.noemoticon.csv.zip"
python -m pip install -r "$SCRIPTPATH/requirements.txt"