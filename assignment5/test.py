import numpy as np
import random
from functools import partial, reduce


def a(x):
    return x**2


def b(x):
    return x*2


func = lambda x: a(b(x))