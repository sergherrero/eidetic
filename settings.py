import os
import dotenv

os.environ.update(dotenv.Dotenv('default.env'))

CATALOG_CONNECTION = os.environ.get('CATALOG_CONNECTION')
CATALOG_QUERY_PATH = os.environ.get('CATALOG_QUERY_PATH')
