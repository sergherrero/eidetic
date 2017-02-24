#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module leverages Locality Sensitive Hashing (LSH) in order to
calculate Approximate Nearest Neighbours and loads the key,
value pairs in-memory (redis).
"""
from __future__ import division

import os
import sys
import numpy as np
import psycopg2
import psycopg2.extras
import nearpy
import logging
import redis
import json
import argparse

from nearpy.filters.nearestfilter import NearestFilter
from nearpy.distances.euclidean import EuclideanDistance

import settings
import core.vector_generator


class ApproxNearestNeighbourEngine:
    """
    Manage the engine to compute approximate neighbours from
    large-dimensional arrays computed using tensorflow.
    The storage is handled in redis.
    """

    def __init__(self, redis_host=settings.REDIS_HOST,
                 redis_port=settings.REDIS.PORT,
                 projection_count=settings.PROJECTION_COUNT,
                 db_url=settings.DB_URL,
                 table_name=settings.IMAGE_TABLE_NAME):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.projection_count = projection_count
        self.db_params = self.db_params
        self.table_name = self.table_name
        self.engine = configure_engine()

    def configure_engine():
        """
        Configures an Approximate Nearest Neighbour calculation
        engine.
        """
        # Create redis storage adapter
        redis_object = redis.Redis(host=self.redis_host,
                                   port=self.redis_port, db=0)
        redis_storage = nearpy.storage.RedisStorage(redis_object)

        # Get hash config from redis
        config = redis_storage.load_hash_configuration('image_hash')

        if config is None:
            lshash = nearpy.hashes.RandomBinaryProjections(
            'image_hash', self.projection_count)
        else:
            lshash = nearpy.hashes.RandomBinaryProjections(None, None)
            lshash.apply_config(config)

        # Create engine
        engine = nearpy.Engine(
            2048, lshashes=[lshash],
            distance=EuclideanDistance(),
            storage=redis_storage)

        # Store the configuration back in redis.
        redis_storage.store_hash_configuration(lshash)
        return engine

    def populate_engine():
        """
        Load redis storage with hashed image vectors.
        """
        self.engine.clean_all_buckets()

        parsed_url = urlparse.urlparse(self.db_url)
        query_params = urlparse.parse_qs(parsed_url.query)
        db_params = "host='{0}' dbname='{1}' user='{2}' password='{3}'".format(
            parsed_url.netloc, parsed_url.path.lstrip("/"),
            query_params['user'][0], query_params['password'][0])

        conn = psycopg2.connect(db_params)
        cursor = conn.cursor(
            'cursor_product_image', cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM {0}'.format(self.table_name))

        for row in cursor:
            r = dict(row)
            r.pop('features', None)
            features = np.asarray(json.loads(row["features"]))
            self.engine.store_vector(features, data=json.dumps(r))
        conn.close()

    def find_candidates_from_image_data(image_data):
        """
        Returns ANN candidates from raw image data.
        """
        from core.vector_generator import ImageFeatureVector
        features = ImageFeatureVector.get_feature_vector(image_data)
        return self.engine.neighbours(features)

    def find_candidates_from_image_url(image_url):
        """
        Returns ANN candidates from an image url.
        """
        image_path = core.vector_generator.download_image(image_url)
        image_data = open(image_path, 'r').read()
        candidates = self.find_product_candidates_from_image_data(image_data)
        os.remove(image_path)
        return candidates


def main(argv):
    parser = argparse.ArgumentParser(
        description="Handle the Approximate Nearest Neighbour Engine.")
    parser.add_argument('--image_url', action='store', required=False, type=str)
    parser.add_argument("--populate", action="store_true", default=False)

    opts = parser.parse_args(argv)
    ann_engine = ApproxNearestNeighbourEngine()

    if opts.populate:
        ann_engine.populate_engine()

    candidates = ann_engine.find_candidates_from_image_url(opts.image_url)

    for (features, data) in candidates:
        sys.stdout.write(data['pdp_url'] + \n)


if __name__ == '__main__':
    main(sys.argv[1:])
