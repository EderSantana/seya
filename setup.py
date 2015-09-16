#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# from setuptools.command.test import test as TestCommand
from setuptools import find_packages
import codecs
import re
# import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# class PyTest(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
#         self.test_suite = True
#
#     def run_tests(self):
#             import pytest
#             errno = pytest.main(self.test_args)
#             sys.exit(errno)

long_description = read('README.rst')


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='seya',
    version='0.1.0',
    description="Deep Learning models and utility functions for Keras",
    long_description=readme + '\n\n' + history,
    author="Eder Santana",
    author_email='edercsjr@gmail.com',
    url='https://github.com/edersantana/seya',
    packages=find_packages(),
    # package_dir={'seya': 'seya'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='seya',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers, Science/Research',
        'License :: The MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    # cmdclass={'test': PyTest},
    extras_require={
        'testing': ['pytest'],
    },
    test_suite='tests',
    tests_require=test_requirements
)
