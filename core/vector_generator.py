#!/usr/bin/env python

"""
This module generates a tensorflow vector from an image (local path or URL) .
"""
from __future__ import division

import os
import re
import numpy as np
import urllib.request
import tempfile

import pandas as pd
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile



class TensorflowSession:
    """
    Singleton class for tensorflow sessions.
    """
    instance = None

    class __TensorflowSession:

        def __init__(self):
            self.create_graph()
            self.tf_session = tf.Session()
            self.next_to_last_tensor = self.tf_session.graph.get_tensor_by_name(
                'pool_3:0')

        def create_graph(self):
            with gfile.FastGFile(os.path.join(
                    './data/imagenet',
                    'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

    def __init__(self):
        if not TensorflowSession.instance:
            TensorflowSession.instance = (
                TensorflowSession.__TensorflowSession())

    def __getattr__(self, name):
        return getattr(self.instance, name)


def extract_feature_vector(input_image_path):
    """
    Extract feature vector using tensorflow
    """
    if input_image_path.startswith("https://") or \
       input_image_path.startswith("http://"):
        image_path = tempfile.mkstemp(suffix='jpg')[1]
        urllib.request.urlretrieve(input_image_path, image_path)
    else:
        image_path = input_image_path

    image_data = gfile.FastGFile(image_path, 'rb').read()
    features = np.squeeze(TensorflowSession().tf_session.run(
        TensorflowSession().next_to_last_tensor,
        {'DecodeJpeg/contents:0': image_data}))

    if input_image_path != image_path:
        os.remove(image_path)

    return features
