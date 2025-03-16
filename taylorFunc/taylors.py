import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Qt5Agg')

import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PolyaCplot.main import polyaStreamplot

def taylor_poly(f_expr, var, n_terms):
    poly = 0
    for k in range(n_terms):
        term = f_expr.diff(var, k).subs(var, 0) / sp.factorial(k) * var**k
        poly += term
    return sp.simplify(poly)


def cos_series(
        x_range=(-5, 5),
        y_range=(-5, 5),
        density=20,
        colormap="plasma"
):
    z = sp.symbols('z')
    f_expr = sp.cos(z)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    plt.sca(ax_left)
    polyaStreamplot(f_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_left.set_title("Функция cos(z)")

    n_init = 5
    taylor_expr = taylor_poly(f_expr, z, n_init)
    plt.sca(ax_right)

    polyaStreamplot(taylor_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_right.set_title(f"Ряд Тейлора cos(z) (n={n_init} членов)")

    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax_slider, 'n', 3 , 20, valinit=n_init, valstep=1)

    def update(val):
        n = int(slider.val)
        ax_right.clear()
        plt.sca(ax_right)
        new_taylor_expr = taylor_poly(f_expr, z, n)
        polyaStreamplot(new_taylor_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
        ax_right.set_title(f"Ряд Тейлора cos(z) (n={n} членов)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def sin_series(
        x_range=(-5, 5),
        y_range=(-5, 5),
        density=20,
        colormap="plasma"
):
    f_expr = sp.sin(z)
    z = sp.symbols('z')

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    plt.sca(ax_left)
    polyaStreamplot(f_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_left.set_title("Функция sin(z)")

    n_init = 5
    taylor_expr = taylor_poly(f_expr, z, n_init)
    plt.sca(ax_right)

    polyaStreamplot(taylor_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_right.set_title(f"Ряд Тейлора sin(z) (n={n_init} членов)")

    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax_slider, 'n', 3 , 20, valinit=n_init, valstep=1)

    def update(val):
        n = int(slider.val)
        ax_right.clear()
        plt.sca(ax_right)
        new_taylor_expr = taylor_poly(f_expr, z, n)
        polyaStreamplot(new_taylor_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
        ax_right.set_title(f"Ряд Тейлора sin(z) (n={n} членов)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def exp_series(
        x_range=(-5, 5),
        y_range=(-5, 5),
        density=20,
        colormap="plasma"
):
    z = sp.symbols('z')
    f_expr = sp.exp(z)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    plt.sca(ax_left)
    polyaStreamplot(f_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_left.set_title("Функция exp(z)")

    n_init = 5
    taylor_expr = taylor_poly(f_expr, z, n_init)
    plt.sca(ax_right)

    polyaStreamplot(taylor_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_right.set_title(f"Ряд Тейлора exp(z) (n={n_init} членов)")

    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax_slider, 'n', 3 , 20, valinit=n_init, valstep=1)

    def update(val):
        n = int(slider.val)
        ax_right.clear()
        plt.sca(ax_right)
        new_taylor_expr = taylor_poly(f_expr, z, n)
        polyaStreamplot(new_taylor_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
        ax_right.set_title(f"Ряд Тейлора exp(z) (n={n} членов)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()