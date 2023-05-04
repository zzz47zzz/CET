import logging
import os

def get_logger(args):
    # create logger
    logger = logging.getLogger("MAIN")
    logger.setLevel(logging.DEBUG)

    # create formatter
    BASIC_FORMAT = "[%(asctime)s]-[%(levelname)s]\t%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    # create consle handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    # create file handler and set level to WARNING
    log_file = os.path.join(args.save_dir, "log")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    print("Log save to %s" % log_file)
    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger