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

        def __init__(self, data_dir=None, model_broadcast=None,
                     node_to_uid_map_broadcast=None,
                     uid_to_label_map_broadcast=None):
            self.data_dir = '' if data_dir is None else data_dir
            self.model_broadcast = model_broadcast
            self.node_to_uid_map_broadcast = node_to_uid_map_broadcast
            self.uid_to_label_map_broadcast = uid_to_label_map_broadcast

            self.create_graph()
            self.tf_session = tf.Session()
            self.next_to_last_tensor = self.tf_session.graph.get_tensor_by_name(
                'pool_3:0')
            self.softmax_tensor = self.tf_session.graph.get_tensor_by_name(
                'softmax:0')
            self.node_to_label = NodeToLabelMap(
                data_dir=self.data_dir,
                node_to_uid_map_broadcast=self.node_to_uid_map_broadcast,
                uid_to_label_map_broadcast=self.uid_to_label_map_broadcast)

        def create_graph(self):
            if model_broadcast is None:
                with gfile.FastGFile(os.path.join(
                        self.data_dir, settings.MODEL_FILE), 'rb') as f:
                    model_data = f.read()
                    graph_def = tf.GraphDef()
            else:
                model_data = model_broadcast.value
            graph_def.ParseFromString(model_data)
            _ = tf.import_graph_def(graph_def, name='')

    def __init__(self, data_dir=None, model_broadcast=None):
        if not Tensorflow.instance:
            Tensorflow.instance = (
                Tensorflow.__Tensorflow(
                    data_dir=data_dir,
                    model_broadcast=model_broadcast,
                    node_to_uid_map_broadcast=node_to_uid_map_broadcast,
                    uid_to_label_map_broadcast=uid_to_label_map_broadcast))

    def __getattr__(self, name):
        return getattr(self.instance, name)


def download_image(input_image_path):
    if input_image_path.startswith("https://") or \
       input_image_path.startswith("http://"):
        image_path = tempfile.mkstemp(suffix='jpg')[1]
        urllib.URLopener.version = settings.USER_AGENT
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
    def get_feature_vector(input_image_path, model_broadcast=None):
        """
        Returns a vector containing the features calculated by
        removing the last layer of the network.
        """
        image_path = download_image(input_image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        tensorflow = Tensorflow(model_broadcast=model_broadcast)
        features = np.squeeze(tensorflow.tf_session.run(
            tensorflow.next_to_last_tensor,
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
        tensorflow = Tensorflow()
        features = np.squeeze(tensorflow.tf_session.run(
            tensorflow.softmax_tensor,
            {'DecodeJpeg/contents:0': image_data}))

        if input_image_path != image_path:
            os.remove(image_path)

        labels = []
        for node_id in features.argsort()[-num_top_predictions:][::-1]:
            labels.append(
                {'score': features[node_id],
                 'tag': tensorflow.node_to_label.id_to_string(node_id)})
        return labels


class NodeToLabelMap(object):
    """
    Converts integer node ID's to a label in english.
    """
    def __init__(self, data_dir=None,
                 node_to_uid_map_broadcast=None,
                 uid_to_label_map_broadcast=None):
        self.data_dir = '' if data_dir is None else data_dir
        self.node_to_uid_map_broadcast = node_to_uid_map_broadcast
        self.uid_to_label_map_broadcast = uid_to_label_map_broadcast

        if node_to_uid_map_broadcast is None:
            uid_lookup_path = os.path.join(
                self.data_dir, settings.UID_TO_LABEL_MAP_FILE)
            if not gfile.Exists(uid_lookup_path):
                raise RuntimeError('File does not exist %s', uid_lookup_path)
            self.node_to_uid_map = gfile.GFile(uid_lookup_path).readlines()
        else:
            self.node_to_uid_map = self.node_to_uid_map_broadcast.value

        if uid_to_label_map_broadcast is None:
            label_lookup_path = os.path.join(
                self.data_dir, settings.NODE_TO_UID_MAP_FILE)
            if not gfile.Exists(label_lookup_path):
                raise RuntimeError('File does not exist %s', label_lookup_path)
            self.uid_to_label_map = gfile.GFile(label_lookup_path).readlines()
        else:
            self.uid_to_label_map = self.uid_to_label_map_broadcast.value

        self.node_to_label = self.load()

    def load(self):
        """
        Loads a label in english for each softmax node.
        """
        uid_to_label = {}
        for line in self.node_to_uid_map:
            parsed_items = re.compile(r'[n\d]*[ \S,]*').findall(line)
            uid_to_label[parsed_items[0]] = parsed_items[2]

        node_id_to_uid = {}
        for line in self.uid_to_label_map:
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
