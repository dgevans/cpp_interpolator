# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:34:48 2013

@author: dgevans
"""

from numpy import *
from cpp_interpolator import interpolate
from cpp_interpolator import interpolate_INFO
from cpp_interpolator import test_INFO
from Spline import Spline

INFO = interpolate_INFO(['spline','hermite'],[10,3],[3,5])

x1 = linspace(0.,1.,10)
x2 = linspace(0,1.,10)
X = Spline.makeGrid([x1,x2])
def f(x):
    return log(1+x[1]) * exp(x[0])
Y = vstack(map(f,X))

fhat = interpolate(X,Y,INFO)

x = linspace(0,1,20).reshape(-1,1)
y = exp(x)
INFO = interpolate_INFO(['hermite'],[19],[3])
ghat = interpolate(x,y,INFO)