#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module vectorizes an entire catalog of product images.
"""
import os
import sys
import json
import logging

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

    def load_product_image_catalog(self, sql_context,
                                   table_name=settings.IMAGE_TABLE_NAME):
        """
        Load product image catalog including vectors, labels and colormap.
        """
        return sql_context.sql("SELECT * FROM %s" % table_name)

    def save_product_image_catalog(self, sql_context,
                                   table_name=settings.IMAGE_TABLE_NAME):
        """
        Save product image catalog including vectors, labels and colormap.
        """
        temp_table_name = '%s_temp' % table_name
        all_columns = sql_context.sql(
            'SELECT * FROM {0}'.format(table_name)).columns

        df = self.df_catalog.select(all_columns)
        df.registerTempTable(temp_table_name)
        #escaped_all_columns = map(lambda s: '`%s`' % s, all_columns)
        sql_context.sql(
            'INSERT INTO TABLE {0} '
            'SELECT {1} FROM {2}'.format(
                table_name,
                ', '.join(all_columns),
                temp_table_name))
        sql_context.catalog.dropTempView(temp_table_name)
        return

def main(argv):

    sql_context = pyspark.sql.SparkSession \
                             .builder \
                             .enableHiveSupport() \
                             .getOrCreate()

    catalog = ProductImageCatalog()
    catalog.load_catalog(sql_context)

    all_categories = (catalog.df_catalog
        .select("category_name")
        .distinct()
        .collect())
    all_categories = [r.category_name for r in all_categories]

    product_image_catalog = catalog.load_product_image_catalog(sql_context)
    existing_categories = (product_image_catalog
        .select("category_name")
        .distinct()
        .collect())
    existing_categories = [r.category_name for r in existing_categories]

    missing_categories = set(all_categories) - set(existing_categories)

    for category_name in missing_categories:
        logging.warn("Adding %s", category_name)
        catalog.load_catalog(sql_context)
        catalog.df_catalog = (catalog.df_catalog
            .filter(catalog.df_catalog.category_name == category_name))
        catalog.add_feature_vector_column(sql_context)
        catalog.add_image_labels_column(sql_context)
        catalog.add_color_palette_column(sql_context)

        catalog.save_product_image_catalog(sql_context)

    sql_context.stop()


if __name__ == '__main__':
    main(sys.argv[1:])
