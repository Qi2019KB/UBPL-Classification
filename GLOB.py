# -*- coding: utf-8 -*-
import os

# Global cache object
global glob_cache
glob_cache = {}


def getValue(key, default=None):
    try:
        return glob_cache[key]
    except KeyError:
        return default


def setValue(key, value):
    glob_cache[key] = value
    return value


root = os.path.abspath(os.path.dirname(__file__))
project = root.split("\\")[-1]

expr = "E:/00Experiment/expr"
temp = "E:/00Experiment/temp"
stat = "E:/00Experiment/statistic"
