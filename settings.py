import os
import dotenv

if os.path.exists('default.env'):
    os.environ.update(dotenv.Dotenv('default.env'))

CATALOG_CONNECTION = os.environ.get('CATALOG_CONNECTION')
CATALOG_QUERY_PATH = os.environ.get('CATALOG_QUERY_PATH')

DATA_DIR = os.environ.get('DATA_DIR')
DB_URL = os.environ.get('DB_URL')
MODEL_FILE = os.environ.get('MODEL_FILE', 'classify_image_graph_def.pb')
NODE_TO_UID_MAP_FILE = os.environ.get(
    'NODE_TO_UID_MAP_FILE', 'imagenet_2012_challenge_label_map_proto.pbtxt')
UID_TO_LABEL_MAP_FILE = os.environ.get(
    'UID_TO_LABEL_MAP_FILE', 'imagenet_synset_to_human_label_map.txt')

USER_AGENT = os.environ.get("USER_AGENT")
