import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Qt5Agg')

import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PolyaCplot.main import polyaStreamplot

def poissChar():
    z = sp.symbols('z')
    f_expr = sp.exp(z)

    polyaStreamplot(f_expr, z, density=20, streamline_density=2.5, x_range=(-2, 2), y_range=(-2, 2))