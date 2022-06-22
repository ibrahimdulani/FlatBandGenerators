import math as mt
import cmath as cmt
import argparse as ap
from math import pi

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from progress.bar import Bar
from numpy.linalg import svd

nu=3
U=3

a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.array([7,8,9])

T=np.zeros((nu*(U+2),nu**2))

for i in range((U+2)*nu):
    if i in range(nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=b[j]
    elif i in range(nu,(U-1)*nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=c[j]
            T[i,(i%nu)+nu*j] += a[j]
    elif i in range((U-1)*nu,U*nu):
        for j in range(nu):
            T[i,(i%nu)+nu*j]=b[j]
    elif i in range(U*nu,(U+1)*nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=a[j]
    else:
        for j in range(nu):
            T[i,(i%nu)+nu*j]=c[j]



