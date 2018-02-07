# -*- coding: utf-8 -*-
"""
Every Logger object will log to global.log in addition to other handlers.

Usage:
    _ = Logger(__name__)  \
        .add_file_handler("logfile.txt", "info")  \
        .create()

The logger object is identical to the Python's built-in Logger object.
"""
import os
import logging

from logging.handlers import RotatingFileHandler


class Logger(object):

    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(os.getcwd(), 'logs', 'global.log')

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(Logger.LOG_FORMAT)

        # Add a File Handler that writes to log/global.log by default
        # The log file will renew itself after 1000 bytes, and save up to
        # 3 backups.
        fh = RotatingFileHandler(Logger.LOG_FILE, maxBytes=1000, backupCount=3)
        fh.setLevel(logging.WARNING)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

    @classmethod
    def levelname_to_int(cls, levelname):
        if levelname == 'debug':
            return logging.DEBUG
        elif levelname == 'info':
            return logging.INFO
        elif levelname == 'warning':
            return logging.WARNING
        elif levelname == 'error':
            return logging.ERROR
        elif levelname == 'critical':
            return logging.CRITICAL
        else:
            raise Exception('Invalid level name %s' % levelname)

    def add_file_handler(self, filename, level):
        """
        Add a logging.RotatingFileHandler to the logger object that will be
        returned by the create() method.
        """
        filepath = os.path.join(os.getcwd(), 'log', filename)
        fh = RotatingFileHandler(filepath, maxBytes=1000, backupCount=3)
        fh.setLevel(Logger.levelname_to_int(level))
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
        return self

    def add_stream_handler(self, level):
        """
        Add a logging.StreamHandler to the logger object that will be returned
        by the create() method.
        """
        sh = logging.StreamHandler()
        sh.setLevel(Logger.levelname_to_int(level))
        sh.setFormatter(self.formatter)
        self.logger.addHandler(sh)
        return self

    def create(self):
        """
        This method MUST be called at the end of Logger class initialization.
        It returns Python's built-in Logger object and allows the use of all
        logging.Logger methods.
        """
        return self.logger
