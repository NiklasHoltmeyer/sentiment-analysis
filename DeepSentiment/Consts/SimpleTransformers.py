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
    'train_batch_size': 40,

    'num_train_epochs': 1,

    'learning_rate': 1.0e-8, 

    'warmup_ratio': 0,#0.2, 1.0e-5,

    "early_stopping_delta" : Training.trainArgs['early_stopping_min'], #1e-3 <- 1 Epoch = this Line isnt doing anything
    "early_stopping_patience" : Training.trainArgs['early_stopping_patience'], #5 <- 1 Epoch = this Line isnt doing anything

    'eval_all_checkpoints': True,

    'silent' : not Global.DEBUG,
    'eval_accumulation_steps' : 8,
    'gradient_accumulation_steps' : 8, # Default 1, The number of training steps to execute before performing a optimizer.step(). Effectively increases the training batch size while sacrificing training time to lower memory consumption,
    'use_cached_eval_features' : False,
    'number_of_training_data_entries' : 800_000, # int or None, None = load all Entries 
    'train_size_ratio' : 0.8,
    'lazy_loading' : False
} #args based on: https://colab.research.google.com/drive/1JKQj-DWHLv_vBdF3VypAIEC6npULOFGy#scrollTo=X7difgOhg664, #https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model 
