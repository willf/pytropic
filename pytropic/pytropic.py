# -*- coding: utf-8 -*-

"""Main module."""


def sliding(string, n):
    for i in range(len(string) - n + 1):
        yield string[i: i+n]
