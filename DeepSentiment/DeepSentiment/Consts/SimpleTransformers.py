from . import (Preprocessing, Training, Global)
MODEL_ARGS = {
    'overwrite_output_dir': True,
    'threshold': 0.5,
    'fp16': False,
    "no_cache": True,
    "no_save" : True,
    "evaluate_during_training" : True,
    "do_lower_case" : True, 
    'reprocess_input_data': True, 
#   'output_dir': 'outputs/',
#   'cache_dir': 'cache/',
    'max_seq_length': Preprocessing.MAX_SEQUENCE_LENGTH, 
    'train_batch_size': 40,#Training.BATCH_SIZE, 

    'num_train_epochs': 1,#Training.EPOCHS, #50

    'learning_rate': 1.0e-8, #Training.Learning_Rate, #1e-3

    'warmup_ratio': 0,#0.2, 1.0e-5,

    "early_stopping_delta" : Training.EARLY_STOPPING_MIN, #1e-3 <- 1 Epoch = this Line isnt doing anything
    "early_stopping_patience" : Training.EARLY_STOPPING_PATIENCE, #5 <- 1 Epoch = this Line isnt doing anything

    'eval_all_checkpoints': True,

    'silent' : not Global.DEBUG,
    'eval_accumulation_steps' : 8,
    'gradient_accumulation_steps' : 8, # Default 1, The number of training steps to execute before performing a optimizer.step(). Effectively increases the training batch size while sacrificing training time to lower memory consumption,
    'use_cached_eval_features' : False
} #args based on: https://colab.research.google.com/drive/1JKQj-DWHLv_vBdF3VypAIEC6npULOFGy#scrollTo=X7difgOhg664, #https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model 
