from DeepSentiment.Consts import (Glove as GLOVE, Paths as PATHS )
from pathlib import Path

sentiFilePath = Path(PATHS.SENTIMENT140_DATASET)
gloveFilePath = Path(GLOVE.GLOVE_FILENAME)

sentiFolder, sentiFileName, sentiURL = sentiFilePath.parent.resolve(), sentiFilePath.name.resolve(), PATHS.SENTIMENT140_URL
gloveFolder, gloveFileName = gloveFilePath.parent.resolve(), gloveFilePath.name.resolve()

#sentiURL = "https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/training.1600000.processed.noemoticon.csv.zip"
#,  = SentimentFolder.resolve(), "training.1600000.processed.noemoticon.csv"

print(sentiFolder, sentiFileName)
print(gloveFolder, gloveFileName)