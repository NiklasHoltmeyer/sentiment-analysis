#region Imports
import argparse
import logging
import consts as CONSTS # GLOBAL, PATHS, GLOVE, TRAINING, PREPROCESSING
from TensorflowHelper import Callbacks, Encoder, Logging
from TensorflowModels import TensorflowModels
#endregion

def str2bool(v):
    #src: https://stackoverflow.com/a/43357954/5026265
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
  
def passArguments(function, args):
    return function(GLOVE = args.glove, 
                CNN_LAYER = args.layer_CNN, 
                POOLING_LAYER = args.layer_POOLING, 
                GRU_LAYER = args.layer_GRU, 
                BiLSTM_Layer = args.layer_BiLSTM, 
                LSTM_Layer = args.layer_LSTM, 
                DENSE_LAYER = args.layer_DENSE,
                logger = logger)
    
def getModelPermutations(prefix=""):
    embeddingLayers = ["--glove", "word2vec"]
    hiddenLayers = [ 
                "--layer_CNN", 
                "--layer_POOLING", 
                "--layer_GRU", 
                "--layer_BiLSTM", 
                "--layer_LSTM", 
                "--layer_DENSE"]

    hiddenLayerCount = len(hiddenLayers)
    hiddenLayerPermutations = 2**hiddenLayerCount - 1

    hiddenPermutations = []
    for permutation in range(hiddenLayerPermutations, 0, -1):   
        flags = []
        for idx, layer in enumerate(hiddenLayers):
            if (permutation & (1 << (idx))):
                flags.append("{} 1".format(layer))
        hiddenPermutations.append(" ".join(flags))

    cmds = []
    for embeddingLayer in embeddingLayers:
        embeddingLayerFlag = embeddingLayer + " 1" if embeddingLayer.startswith("--") else ""
        for hiddenPermutation in hiddenPermutations:
            cmd = "{} {} {}".format(prefix, embeddingLayerFlag, hiddenPermutation).replace("  ", " ")
            cmds.append(cmd)
    return cmds
    
parser = argparse.ArgumentParser()

parser.add_argument("mode", choices=['train', 'validate', 'permutations'], help="[train] = train model, \n [validate] = test model against validation data, [permutations] = get all possible models to train \n")

parser.add_argument('--glove', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Should the Model use Glove or should it train its own WordVec")

parser.add_argument('--layer_CNN',         type=str2bool, nargs='?', const=True, default=False, help="Add a CNN Layer")
parser.add_argument('--layer_POOLING',     type=str2bool, nargs='?', const=True, default=False, help="Add a POOLING Layer")
parser.add_argument('--layer_GRU',         type=str2bool, nargs='?', const=True, default=False, help="Add a GRU Layer")
parser.add_argument('--layer_BiLSTM',      type=str2bool, nargs='?', const=True, default=False, help="Add a BiLSTM Layer")
parser.add_argument('--layer_LSTM',        type=str2bool, nargs='?', const=True, default=False, help="Add a LSTM Layer")
parser.add_argument('--layer_DENSE',       type=str2bool, nargs='?', const=True, default=False, help="Add a DENSE Layer")
    
args = parser.parse_args()
print(args)
print()
loggingPrefix = "[{}]".format(args.mode) #train, validate

loggingFile = Logging.createLogPath(GLOVE = args.glove, 
            CNN_LAYER = args.layer_CNN, 
            POOLING_LAYER = args.layer_POOLING, 
            GRU_LAYER = args.layer_GRU, 
            BiLSTM_Layer = args.layer_BiLSTM, 
            LSTM_Layer = args.layer_LSTM, 
            DENSE_LAYER = args.layer_DENSE,
            PREFIX = loggingPrefix)

logger = Logging.getLogger(loggingFile = loggingFile, consoleLogging = True, logginLevel = logging.DEBUG)
#loadModel testModel

           
if("validate" in args.mode):
    print("VALIDATE")
    model = passArguments(TensorflowModels().loadModel, args)
elif ("train" in args.mode):
    print("TRAIN")
    model, history = passArguments(TensorflowModels().trainModel, args)

elif ("permutations" in args.mode):
    for model in getModelPermutations(prefix="python train.py train "):
        print(model)





