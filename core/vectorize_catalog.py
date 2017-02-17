#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module vectorizes an entire catalog of product images.
"""
import os
import sys
import json

import pyspark.sql
from pyspark.sql import Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

import settings
import core.vector_generator
from core.vector_generator import ImageFeatureVector


class ProductImageCatalog:
    """
    Stores information about the product catalog including
    name, categories, URL and feature vectors.
    """
    def __init__(self, catalog_name=None,
                 catalog_connection=settings.CATALOG_CONNECTION,
                 catalog_query_path=settings.CATALOG_QUERY_PATH):

        self.catalog_name = catalog_name
        self.catalog_connection = catalog_connection

        if not self.catalog_connection.startswith('jdbc:'):
            self.catalog_connection = 'jdbc:' + self.catalog_connection
        self.catalog_query = open(catalog_query_path).read().rstrip()


    def load_catalog(self, sql_context):
        """
        Load image catalog information into a pyspark dataframe.
        (id, image_url, product_id, product_name, product_url, category_name)
        """
        self.df_catalog = (sql_context.read
            .format('jdbc')
            .options(url=self.catalog_connection,
                     dbtable="(" + self.catalog_query + ") as foo",
                     driver="org.postgresql.Driver")
            .load())

    def add_color_palette_column(self, sql_context):
        """
        Add column to the catalog containing the color palette using ColorThief.
        """

        def _get_color_palette(url):
            import colorthief
            image_path = core.vector_generator.download_image(url)
            palette = colorthief.ColorThief(image_path).get_palette(
                color_count=6, quality=1)
            if url != image_path:
                os.remove(image_path)
            return json.dumps(palette)

        self.df_catalog = (self.df_catalog
            .withColumn(
                'palette', F.udf(_get_color_palette, T.StringType())(
                    self.df_catalog.image_url)))

    def add_feature_vector_column(self, sql_context):
        """
        Add column to the catalog containing the feature vector
        obtained from Tensorflow.
        """

        def _get_feature_vector(url):
            features = ImageFeatureVector.get_feature_vector(url).tolist()
            return json.dumps(features)

        self.df_catalog = (self.df_catalog
            .withColumn(
                'features', F.udf(_get_feature_vector, T.StringType())(
                    self.df_catalog.image_url)))

    def add_image_labels_column(self, sql_context):
        """
        Add column to the catalog with the labels obtained by Tensorflow.
        """
        def _get_image_labels(url):
            labels = ImageFeatureVector.get_image_labels(url)[0]
            labels['score'] = str(labels['score'])
            return json.dumps(labels)

        self.df_catalog = (self.df_catalog
            .withColumn(
                'image_labels', F.udf(_get_image_labels, T.StringType())(
                    self.df_catalog.image_url)))
