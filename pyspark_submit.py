#!/usr/bin/env python
"""
The goal of this module is to wrap all the configuration required
to launch a spark job. Additionally it will capture stdout and allow
controlling logging.
"""
from __future__ import division

import os
import sys
import zipfile
import tempfile
import subprocess
import glob
import shutil
import logging
import argparse



def get_temp_file_path(suffix):
    """
    Return the name for a a temporary file.
    """
    tf = tempfile.NamedTemporaryFile(suffix=suffix)
    temp_file_name = tf.name
    tf.close()
    return temp_file_name


def create_zip_file_from_path(source_path):
    """
    Generates a zip file containing all .py files in folders and subfolders.
    """
    os.chdir(source_path)
    zip_filename = get_temp_file_path(".zip")
    zip_file = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

    for (path, dirs, files) in os.walk(source_path):
        for filename in files:
            if filename.endswith(".py"):
                rel_path = os.path.relpath(path, source_path)
                full_filename = os.path.join(rel_path, filename)
                logging.info("Adding to zip file {0}".format(full_filename))
                zip_file.write(full_filename)
    zip_file.close()
    return zip_filename


PACKAGES_TO_REMOVE = ("easy_install", "pip", "pkg_resources",
                      "setuptools", "wheel")


def generate_packages_zip_file(source_path, requirements_filepath):
    """
    Generates a zip file containing all dependencies that will be shipped
    to the spark worker nodes.
    """
    # Create virtualenv
    temp_dir = tempfile.mkdtemp()
    os.chdir(source_path)
    virtualenv_cmd = "/usr/bin/virtualenv {0}/venv".format(temp_dir)
    logging.info("Running: {}".format(virtualenv_cmd))
    subprocess.check_output(virtualenv_cmd, shell=True)

    # Install pip packages from requirements file.
    pip_cmd = "{0}/venv/bin/pip install -r {1}".format(
        temp_dir, requirements_filepath)
    logging.info("Running: {}".format(pip_cmd))
    subprocess.check_output(pip_cmd, shell=True)

    # Remove unnecessary packages
    path_venv = temp_dir + "/venv/lib/python2.7/site-packages/"
    for package in PACKAGES_TO_REMOVE:
        for path in glob.glob(path_venv + package + "*"):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    return create_zip_file_from_path(path_venv)


def prepare_spark_zip_files(source_path, requirements_file):
    """
    Returns a tuple with the absolute paths to the
    zip files containing both the source py files and the packages.
    """
    return (create_zip_file_from_path(source_path),
            generate_packages_zip_file(source_path, requirements_file))


DEFAULT_DRIVER_CLASS_PATH = [
    "/usr/lib/hadoop-lzo/lib/*",
    "/usr/lib/hadoop/hadoop-aws.jar",
    "/usr/share/aws/aws-java-sdk/*",
    "/usr/share/aws/emr/emrfs/conf",
    "/usr/share/aws/emr/emrfs/lib/*",
    "/usr/share/aws/emr/emrfs/auxlib/*",
    "/usr/share/aws/emr/security/conf",
    "/usr/share/aws/emr/security/lib/*"]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Script to submit spark jobs.")
    parser.add_argument('--file_dir', action='store', required=False,
                        type=str, default="files",
                        help='directory of data files to be shipped to the workers.')
    parser.add_argument("--requirements_file", action='store', required=False,
                        type=str, default="requirements.txt",
                        help='packages to install in the worker nodes.')
    parser.add_argument('--jar_dir', action='store', required=False,
                        type=str, default="bin",
                        help='directory containing jar files.')
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help='print spark-submit command.')
    parser.add_argument('args', nargs='*',
                        help='script containing the job to execute and its arguments.')
    opts = parser.parse_args(argv)

    # Include auxiliary files.
    if os.path.exists(opts.file_dir) and len(os.listdir(opts.file_dir)) > 1:
        files = map(lambda f: os.path.join(opts.files, f),
                    os.listdir(opts.file_dir))
        files_flag = "--files {0}".format(",".join(files))
    else:
        files_flag = ""

    # Include source python files and package files.
    current_abs_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(__file__)
    py_files = prepare_spark_zip_files(
        "." if current_dir == "" else current_dir, opts.requirements_file)
    py_files_flag = "--py-files " + ",".join(py_files)
    os.chdir(os.path.dirname(current_abs_path))

    # Include jar files (e.g database connectors)
    if os.path.exists(opts.jar_dir) and len(os.listdir(opts.jar_dir)) > 1:
        jars = map(lambda f: os.path.join(opts.jar_dir, f),
                   os.listdir(opts.jar_dir))
        jars_flag = "--jars {0}".format(",".join(jars))
    else:
        jars_flag = ""

    # Specify driver class-path
    driver_class_path_flag = "--driver-class-path " + ":".join(
        (os.listdir(opts.jar_dir) if os.path.exists(opts.jar_dir) else []) +
        DEFAULT_DRIVER_CLASS_PATH)

    cmd = ("/usr/bin/spark-submit {0} {1} {2} {3} {4}".format(
        files_flag, py_files_flag, jars_flag, driver_class_path_flag,
        ' '.join(opts.args)))

    if opts.dry_run:
        sys.stdout.write(cmd)
    else:
        subprocess.check_output(cmd, shell=True)

    # Clean shipped zip files.
    for file_to_remove in py_files:
        os.remove(file_to_remove)

if __name__ == '__main__':
    main(sys.argv[1:])
