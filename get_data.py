import wget
import os
import logging

from config import *

if __name__ == '__main__':
    log = logging.getLogger(__file__)

    if not os.path.exists(DATA_RAW_PATH):
        os.mkdir(DATA_RAW_PATH)
    try:
        wget.download(DATA_URL, out=FILE_RAW_PATH)
    except:
        log.error("file exist in disk")
