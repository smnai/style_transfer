# Standard Python
import os

SRC_PATH, _ = os.path.split(os.path.abspath(__file__))
BASE_DIR = os.path.join(SRC_PATH, '..')
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DEBUG = os.environ.get('DEBUG', False)
