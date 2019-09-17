#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:58:46 2019

@author: dell
"""

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

mnist=fetch_mldata('MNIST original')
print(mnist)