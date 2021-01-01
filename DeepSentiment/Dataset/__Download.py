import zipfile
from os import listdir, remove
from os.path import isfile, join
from pathlib import Path
import shutil

import pandas as pd
import csv

import chakin

from DeepSentiment.Consts import (Glove as GLOVE, Paths as PATHS )

def downloadGlove(gloveFolderPath, gloveDim = 100): #<- downloads every Dimension, so the indx doesnt really matter
    print("Download Glove")
    gloveTwitterIDX = {  '25' : 17, "50" : 18, "100" : 19, "200" : 20 } #key = dim, value = index
    chakinIDX = gloveTwitterIDX[str(gloveDim)]
    zipFile = chakin.download(number=chakinIDX, save_dir='./tmp/glove')
    print("Unzip Glove")
    ##unzip
    unzipedPath = "./tmp/glove_unzipped"
    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
        zip_ref.extractall(unzipedPath)
        
    ##Move
    dest = Path(gloveFolderPath)
    destAbsolute = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    for f in [join(unzipedPath, f) for f in listdir(unzipedPath)]:
        shutil.move(f, destAbsolute)
        
    ##Try to delete TMP
    try:
        Path("./tmp/").unlink()
    except:
        pass
    
def downloadSentiment140(zipUrl, folder, file):
    Path(folder).mkdir(parents=True, exist_ok=True)
    sentiPath = Path(folder, file)
    print("Download Senti140")
    sentiDF = pd.read_csv(zipUrl, compression="zip", encoding ="ISO-8859-1" , names=["sentiment", "ids", "date", "flag", "user", "text"])
    sentiDF.to_csv(sentiPath.resolve(), encoding ="ISO-8859-1", index=False, header=False, quoting=csv.QUOTE_ALL)      
    

sentiFilePath = Path(PATHS.SENTIMENT140_DATASET)
gloveFilePath = Path(GLOVE.GLOVE_FILENAME)

sentiFolder, sentiFileName, sentiURL = sentiFilePath.parent, sentiFilePath.name, PATHS.SENTIMENT140_URL
gloveFolder = gloveFilePath.parent

if not gloveFolder.exists():
    downloadGlove(gloveFolder)

if not sentiFolder.exists():
    downloadSentiment140(sentiURL, sentiFolder, sentiFileName)
    
