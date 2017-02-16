#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module vectorizes an entire catalog of product images.
"""
import os

from pyspark.sql import Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

import settings
import core.vector_generator


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
        (id, image_url, product_name, category_name)
        """
        self.df_catalog = (sql_context.read
            .format('jdbc')
            .options(url=self.catalog_connection,
                     dbtable="(" + self.catalog_query + ") as foo",
                     driver="org.postgresql.Driver")
            .load())


    def add_feature_vector_column(self, sql_context):
        """
        Add column to the catalog representing the feature vector
        obtained from Tensorflow.
        """
        # Add column with feature vector
        # Add column with image tags
        raise RuntimeError("Not implemented yet")
