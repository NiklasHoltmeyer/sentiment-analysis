trainArgs = {
    'number_of_training_data_entries' : None, # int or None, None = load all Entries 
    'num_train_epochs' : 1,
    'learning_rate' : 1e-3,
    'early_stopping_min' : 1e-3, #early_stopping_delta
    'early_stopping_patience' : 5,    
    'train_batch_size' : 1024,    
    'train_size_ratio' : 0.8, # split train, test
}


mapKeys = { #used for Key synonymous e.q  num_train_epochs = epochs
    "early_stopping_delta" : "early_stopping_min",   
    "epochs" : 'num_train_epochs',
    'lr' : "learning_rate",
    'batch_size' : 'train_batch_size',
    'trainingEntries' : 'number_of_training_data_entries',
    'entries' : 'number_of_training_data_entries'
}


