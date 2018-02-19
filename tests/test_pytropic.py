#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pytropic` package."""


import unittest

from pytropic import pytropic


class TestPytropic(unittest.TestCase):
    """Tests for `pytropic` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_sliding(self):
        assert(list(pytropic.sliding("test", 2)) == ['te', 'es', 'st'])
