#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: Put package requirements here
]

setup_requirements = [
    # TODO(willf): Put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: Put package test requirements here
]

setup(
    name='pytropic',
    version='0.2.0',
    description="Train and predict string entropy based on character n-grams",
    long_description=readme + '\n\n' + history,
    author="Will Fitzgerald",
    author_email='will.fitzgerald@gmail.com',
    url='https://github.com/willf/pytropic',
    packages=find_packages(include=['pytropic']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='pytropic',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
