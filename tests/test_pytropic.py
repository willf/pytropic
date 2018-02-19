#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pytropic` package."""


import unittest
import os
from pytropic import pytropic


class TestPytropic(unittest.TestCase):
    """Tests for `pytropic` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_sliding(self):
        assert(list(pytropic.sliding("test", 2)) == ['te', 'es', 'st'])

    def test_read(self):
        m = pytropic.Model(2)
        cwd = os.path.dirname(os.path.realpath(__file__))
        p = os.path.join(cwd, "data", "google_books_2.tsv")
        with open(p) as f:
            m.read(f)
        assert(m.entropy("fish") > 0)
