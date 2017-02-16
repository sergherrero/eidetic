import os
import dotenv

os.environ.update(dotenv.Dotenv('default.env'))

CATALOG_CONNECTION = os.environ.get('CATALOG_CONNECTION')
CATALOG_QUERY_PATH = os.environ.get('CATALOG_QUERY_PATH')

DATA_DIR = os.environ.get('DATA_DIR')
MODEL_FILE = os.environ.get('MODEL_FILE')
NODE_TO_UID_MAP_FILE = os.environ.get('NODE_TO_UID_MAP_FILE')
UID_TO_LABEL_MAP_FILE = os.environ.get('UID_TO_LABEL_MAP_FILE')
