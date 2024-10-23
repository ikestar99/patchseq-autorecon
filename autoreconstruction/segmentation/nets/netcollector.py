#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import torch.nn as nn
import importlib


class NetCollector(object):
    """
    Collects the different neural network architectures into a library
    """
    module_list = dict()

    @classmethod
    def add_module(
            cls,
            module: nn.Module,
            identifier: str
    ):
        NetCollector.module_list[identifier] = module

    @classmethod
    def get_module(
            cls,
            identifier: str
    ):
        importlib.import_module("neurotorch.nets." + identifier)
        return NetCollector.module_list[identifier]
