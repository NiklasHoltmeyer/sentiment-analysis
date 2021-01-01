import logging

def defaultLogger(level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'):
    logging.basicConfig(level=level, format=format)

    logger = logging.getLogger("DeepSentiment")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("nltk_data").setLevel(logging.WARNING)
    
    return logger