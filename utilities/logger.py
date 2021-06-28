"""
Created by ingridtveten at 29/09/2020
@author: Ingrid Tveten
"""
from os import makedirs, path
import logging
from logging.handlers import RotatingFileHandler


MAX_LOGSIZE_MEGABYTE = 10   # Max size of logfile before a new is created
LOGFILE_BACKUPCOUNT = 10     # Number of logfiles created before overwriting

# Path where the logfiles will be stored
LOGFILE_PATH = path.join('.', 'logs')


""" Set console logging level """
log_level = logging.INFO


def init_logger(logger_name="default_logger"):

    base_logger = create_base_logger()

    filehandler = create_filehandler(logger_name)
    base_logger.addHandler(filehandler)

    consolehandler = create_consolehandler(level=log_level)
    base_logger.addHandler(consolehandler)

    base_logger.info("%s logger initiated", logger_name)


def create_base_logger():
    """
    Create base logger that log all messages to lowest level
    """
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    return logger


def create_filehandler(logger_name):
    """
    Create application rotating file handler
    :param logger_name: Logfile name
    """
    filename = logger_name + ".log"
    if not path.isdir(LOGFILE_PATH):
        makedirs(LOGFILE_PATH)
    logfile_path = path.join(LOGFILE_PATH, filename)

    handler = RotatingFileHandler(logfile_path,
                                  maxBytes=MAX_LOGSIZE_MEGABYTE * 1024 * 1024,
                                  backupCount=LOGFILE_BACKUPCOUNT,
                                  encoding="utf-8")

    handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter("\t".join(("%(asctime)s", "%(threadName)s",
                                "%(levelname)s", "%(module)s", "%(message)s")))
    handler.setFormatter(formatter)

    return handler


def create_consolehandler(level=logging.INFO):
    """
    Create console handler 
    :param level: Optional logging level, default INFO
    """
    handler = logging.StreamHandler()
    handler.setLevel(level)
    return handler


def close_logger():
    """
    Perform logging cleanup actions. To be called at application exit.
    """
    logging.shutdown()
    base_logger = logging.getLogger()
    base_logger.info("Closing logger")
    base_logger.handlers = []  # Clear list of handlers

