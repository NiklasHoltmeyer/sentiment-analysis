mkdir -p ../data/sentiment140
wget -nc https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/../data/training.1600000.processed.noemoticon.csv.zip -P ../data/sentiment140
unzip -n -d ../data/sentiment140/ ../data/sentiment140/training.1600000.processed.noemoticon.csv.zip

mkdir -p ../data/glove
wget -nc http://nlp.stanford.edu/../data/glove.twitter.27B.zip -P ../data/glove
unzip -n -d ../data/glove ../data/glove/glove.twitter.27B.zip

python -m pip install -r requirements.txt