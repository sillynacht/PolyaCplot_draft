import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Union, Callable


def gridForImage(
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        density: int,
        linspace_kwargs: dict | None = None,
        meshgrid_kwargs: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    get a coordinate grid for an image

    :param x_range: Range of x-axis values, default is (-2, 2).
    :param y_range: Range of y-axis values, default is (-2, 2).
    :param density: Number of grid points per axis for vector field.
    :param linspace_kwargs: parameters for np.linspace.
    :param meshgrid_kwargs: parameters for np.meshgrid.
    :return: grid coordinates for real and imaginary parts.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    if linspace_kwargs is None:
        linspace_kwargs = {}
    if meshgrid_kwargs is None:
        meshgrid_kwargs = {}

    x = np.linspace(x_min, x_max, density, **linspace_kwargs)
    y = np.linspace(y_min, y_max, density, **linspace_kwargs)

    X, Y = np.meshgrid(x, y, **meshgrid_kwargs)

    return X, Y


def getFunc(f_expr: Union[sp.Expr, Callable[[np.ndarray], np.ndarray]], z: sp.Symbol) -> Callable:
    if callable(f_expr):
        return f_expr
    else:
        return sp.lambdify(z, f_expr, "numpy")


def polyaVectorplot(
        f_expr: Union[sp.Expr, Callable[[np.ndarray], np.ndarray]],
        z: sp.Symbol,
        x_range: tuple[float, float] = (-2, 2),
        y_range: tuple[float, float] = (-2, 2),
        density: int = 10,
        colormap: str = "plasma",
        quiver_kwargs: dict = None,
        linspace_kwargs: dict = None,
        meshgrid_kwargs: dict = None
) -> None:
    """
    Plots the Polya vector field for a complex function f(z).

    :param f_expr: Expression representing the complex function f(z).
    :param z: Symbol representing the complex variable z.
    :param x_range: Range of x-axis values, default is (-2, 2).
    :param y_range: Range of y-axis values, default is (-2, 2).
    :param density: Number of grid points per axis for vector field.
    :param colormap: Color of the streamplot, default is "plasma".
    :param linspace_kwargs: parameters for np.linspace.
    :param meshgrid_kwargs: parameters for np.meshgrid.
    :param quiver_kwargs: parameters for plt.quiver.
    """
    X, Y = gridForImage(x_range, y_range, density, linspace_kwargs, meshgrid_kwargs)
    Z = X + Y * 1j

    func = getFunc(f_expr, z)
    result = func(Z)

    u, v = result.real, -result.imag
    magnitude = np.sqrt(u ** 2 + v ** 2)
    U = u / (magnitude + 1e-9)
    V = v / (magnitude + 1e-9)

    if quiver_kwargs is None:
        quiver_kwargs = {}

    plt.quiver(X, Y, U, V, magnitude, cmap=colormap, **quiver_kwargs)

def polyaStreamplot(
        f_expr: sp.Expr,
        z: sp.Symbol,
        x_range: tuple[float, float] = (-2, 2),
        y_range: tuple[float, float] = (-2, 2),
        density: int = 10,
        streamline_density: Union[tuple[int, int], int] = (2, 2),
        colormap: str = "plasma",
        streamplot_kwargs: dict = None,
        linspace_kwargs: dict = None,
        meshgrid_kwargs: dict = None
):
    """
    Plots the streamplot for a complex function f(z) by Polya vector field.

    :param f_expr: Expression representing the complex function f(z).
    :param z: Symbol representing the complex variable z.
    :param x_range: Range of x-axis values, default is (-2, 2).
    :param y_range: Range of y-axis values, default is (-2, 2).
    :param density: Number of grid points per axis for streamplot.
    :param streamline_density: Density of the streamlines, default is (2, 2).
    :param colormap: Color of the streamplot, default is "plasma".
    :param streamplot_kwargs: parameters for plt.streamplot.
    :param linspace_kwargs: parameters for np.linspace.
    :param meshgrid_kwargs: parameters for np.meshgrid.
    """
    X, Y = gridForImage(x_range, y_range, density, linspace_kwargs, meshgrid_kwargs)
    Z = X + Y * 1j

    func = sp.lambdify(z, f_expr)
    result = func(Z)

    u, v = result.real, -result.imag
    magnitude = np.sqrt(u ** 2 + v ** 2)

    if streamplot_kwargs is None:
        streamplot_kwargs = {}

    plt.streamplot(X, Y, u, v, color=magnitude, cmap=colormap, density=streamline_density, **streamplot_kwargs)


def zeros(
        f_expr: sp.Expr,
        z: sp.Symbol,
        x_range: tuple[float, float] = (-2, 2),
        y_range: tuple[float, float] = (-2, 2),
        scatter_kwargs: dict = None
):
    """
    highlights zeros in the plot

    :param f_expr: Expression representing the complex function f(z).
    :param z: Symbol representing the complex variable z.
    :param x_range: Range of x-axis values, default is (-2, 2).
    :param y_range: Range of y-axis values, default is (-2, 2).
    :param scatter_kwargs: parameters for plt.scatter.
    """

    if not isinstance(f_expr, sp.Expr):
        raise TypeError("Zero detection is supported only for sympy expressions.")

    zeros_sym = sp.solve(f_expr, z)
    zeros = []

    for zero in zeros_sym:
        zero_val = sp.N(zero)
        re_zero = float(sp.re(zero_val))
        im_zero = float(sp.im(zero_val))

        if x_range[0] <= re_zero <= x_range[1] and y_range[0] <= im_zero <= y_range[1]:
            zeros.append((re_zero, -im_zero))

    if zeros:
        zeros_re, zeros_im = zip(*zeros)
        if scatter_kwargs is None:
            scatter_kwargs = {}
        plt.scatter(zeros_re, zeros_im, color='red', s=100, zorder=2, label="Нули", **scatter_kwargs)


def poles(
        f_expr: sp.Expr,
        z: sp.Symbol,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        scatter_kwargs: dict = None
):
    """
    highlights poles in the plot

    :param f_expr: Expression representing the complex function f(z).
    :param z: Symbol representing the complex variable z.
    :param x_range: Range of x-axis values, default is (-2, 2).
    :param y_range: Range of y-axis values, default is (-2, 2).
    :param scatter_kwargs: parameters for plt.scatter.
    """
    if not isinstance(f_expr, sp.Expr):
        raise TypeError("Pole detection is supported only for sympy expressions.")

    f_expr_simpl = sp.together(sp.simplify(f_expr))
    num, den = sp.fraction(f_expr_simpl)
    poles_sym = sp.solve(den, z)
    poles = []

    for sol in poles_sym:
        sol_val = sp.N(sol)
        re_val = float(sp.re(sol_val))
        im_val = float(sp.im(sol_val))

        if x_range[0] <= re_val <= x_range[1] and y_range[0] <= im_val <= y_range[1]:
            poles.append((re_val, im_val))

    if poles:
        poles_re, poles_im = zip(*poles)
        if scatter_kwargs is None:
            scatter_kwargs = {}
        plt.scatter(poles_re, poles_im, color='blue', s=100, marker='x', label="Полюса", **scatter_kwargs)


def deformedCoordinateGrid(
        f_expr,
        z,
        x_range: tuple[float, float] = (-5, 5),
        y_range: tuple[float, float] = (-5, 5),
        density: int = 30,
        grid_levels: int = 10
):
    X, Y = gridForImage(x_range, y_range, density)
    Z = X + Y * 1j

    func = sp.lambdify(z, f_expr)
    result = func(Z)

    u, v = result.real, -result.imag

    # TODO доделать
    # plt.contour(u, v, X, levels=grid_levels, colors='grey')
    # plt.contour(u, v, Y, levels=grid_levels, colors='grey')


if __name__ == '__main__':
    fig, ax = plt.subplots()

    z = sp.symbols('z')
    f_expr = 1 / z ** 2 
    polyaStreamplot(f_expr, z, density=20, streamline_density=2.5, x_range=(-5, 5), y_range=(-5, 5))
    deformedCoordinateGrid(f_expr, z)
    plt.legend()
    plt.show()


    polyaVectorplot(f_expr, density=20)
    plt.show()