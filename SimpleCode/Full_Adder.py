# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:48:51 2024

@author: casas
"""

import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def Calc_Half(A, B):
    S = XOR(A, B)
    C = AND(A, B)
    print(S, C)

def Calc_Full(A, B, C_IN):
    S1 = XOR(A, B)
    S  = XOR(S1, C_IN)

    C1 = AND(A, B)
    C2 = AND(S1, C_IN)
    C  = OR(C1, C2)

    print(S, C)


Calc_Full(0, 0, 0)
Calc_Full(0, 0, 1)
Calc_Full(0, 1, 0)
Calc_Full(0, 1, 1)
Calc_Full(1, 0, 0)
Calc_Full(1, 0, 1)
Calc_Full(1, 1, 0)
Calc_Full(1, 1, 1)
