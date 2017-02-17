#!/usr/bin/env python

"""
This module generates tensorflow vector and labels from an image
(local path or URL).
"""
from __future__ import division

import os
import re
import numpy as np
import urllib
import tempfile

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile

import settings


class Tensorflow:
    """
    Singleton class for tensorflow sessions.
    """
    instance = None

    class __Tensorflow:

        def __init__(self, data_dir=None):
            self.data_dir = '' if data_dir is None else data_dir
            self.create_graph()
            self.tf_session = tf.Session()
            self.next_to_last_tensor = self.tf_session.graph.get_tensor_by_name(
                'pool_3:0')
            self.softmax_tensor = self.tf_session.graph.get_tensor_by_name(
                'softmax:0')
            self.node_to_label = NodeToLabelMap(data_dir=data_dir)

        def create_graph(self):
            with gfile.FastGFile(os.path.join(
                    self.data_dir, settings.MODEL_FILE), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

    def __init__(self, data_dir=None):
        if not Tensorflow.instance:
            Tensorflow.instance = (
                Tensorflow.__Tensorflow(data_dir=data_dir))

    def __getattr__(self, name):
        return getattr(self.instance, name)


def download_image(input_image_path):
    if input_image_path.startswith("https://") or \
       input_image_path.startswith("http://"):
        image_path = tempfile.mkstemp(suffix='jpg')[1]
        urllib.urlretrieve(input_image_path, image_path)
    else:
        image_path = input_image_path
    return image_path


class ImageFeatureVector:
    """
    Stores information about the image and generates
    the tensorflow vector.
    """
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path
        self.features = self.get_feature_vector(self.input_image_path)
        self.labels = self.get_image_labels(self.input_image_path)


    @staticmethod
    def get_feature_vector(input_image_path):
        """
        Returns a vector containing the features calculated by
        removing the last layer of the network.
        """
        image_path = download_image(input_image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        features = np.squeeze(Tensorflow().tf_session.run(
            Tensorflow().next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data}))

        if input_image_path != image_path:
            os.remove(image_path)
        return features


    @staticmethod
    def get_image_labels(input_image_path, num_top_predictions=1):
        """
        Returns a list with the scores and tags of the nodes that best
        match the input image.
        """
        image_path = download_image(input_image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        features = np.squeeze(Tensorflow().tf_session.run(
            Tensorflow().softmax_tensor,
            {'DecodeJpeg/contents:0': image_data}))

        if input_image_path != image_path:
            os.remove(image_path)

        labels = []
        for node_id in features.argsort()[-num_top_predictions:][::-1]:
            labels.append(
                {'score': features[node_id],
                 'tag': Tensorflow().node_to_label.id_to_string(node_id)})
        return labels


class NodeToLabelMap(object):
    """
    Converts integer node ID's to a label in english.
    """
    def __init__(self, data_dir=None):
        self.data_dir = '' if data_dir is None else data_dir
        label_lookup_path = os.path.join(
            self.data_dir, settings.NODE_TO_UID_MAP_FILE)

        uid_lookup_path = os.path.join(
            self.data_dir, settings.UID_TO_LABEL_MAP_FILE)
        self.node_to_label = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """
        Loads a label in english for each softmax node.
        """
        if not gfile.Exists(uid_lookup_path):
            raise RuntimeError('File does not exist %s', uid_lookup_path)

        if not gfile.Exists(label_lookup_path):
            raise RuntimeError('File does not exist %s', label_lookup_path)

        uid_to_label = {}
        for line in gfile.GFile(uid_lookup_path).readlines():
            parsed_items = re.compile(r'[n\d]*[ \S,]*').findall(line)
            uid_to_label[parsed_items[0]] = parsed_items[2]

        node_id_to_uid = {}
        for line in gfile.GFile(label_lookup_path).readlines():
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        node_id_to_label = {}
        for (node_id, uid) in node_id_to_uid.items():
            if uid not in uid_to_label:
                raise RuntimeError('Failed to locate: %s', uid)
            node_id_to_label[node_id] = uid_to_label[uid]
        return node_id_to_label

    def id_to_string(self, node_id):
        if node_id not in self.node_to_label:
            return ''
        return self.node_to_label[node_id]
