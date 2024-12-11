"""
Python odule for generating tapers.
"""

import numpy as np
from geometry import _mirror, _euler_curve

def linear_taper(w0:float, w1:float, length:float, num_pts=2) -> np.ndarray[float]:
    """
    Generates vertices for linear taper in clockwise order.

    Parameters
    ----------
    w0 : float
        Width of one end of the taper.
    w1 : float
        Width of the other end of the taper.
    length : float
        Distance between both ends of the taper.
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if num_pts < 2:
        raise ValueError(f'Input parameter <num_pts> must be at least 2, but was given a value of {num_pts}.')
    if num_pts % 2 != 0:
        num_pts += 1

    slope = (w1-w0)/2 / length
    x = np.linspace(0, length, num_pts)
    y = slope*x + w0/2

    vertices = _mirror(x=x, y=y, axis='y')

    return vertices

def parabolic_taper(w0:float, w1:float, length:float, num_pts=100) -> np.ndarray[float]:
    """
    Generates vertices for parabolic taper in clockwise order.

    Parameters
    ----------
    w0 : float
        Width of one end of the taper.
    w1 : float
        Width of the other end of the taper.
    length : float
        Distance between both ends of the taper.
    num_pts : int
        Number of points on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if num_pts % 2 != 0:
        num_pts += 1

    a = 4 * length / (w1**2 - w0**2)
    c = a * w0**2 / 4

    x = np.linspace(0, length, num_pts)
    y = np.sqrt((x + c) / a)

    # Concatenate top and bottom vertices
    verts_top = np.vstack((x,y)).T
    verts_bot = np.vstack((x,-y)).T
    verts_bot = verts_bot[::-1]
    vertices = np.vstack((verts_top, verts_bot))
    
    return vertices

def gaussian_taper(w0:float, w1:float, length:float, num_pts=100) -> np.ndarray[float]:
    """
    Generates vertices for Gaussian taper in clockwise order.

    Parameters
    ----------
    w0 : float
        Width of one end of the taper.
    w1 : float
        Width of the other end of the taper.
    length : float
        Distance between both ends of the taper.
    num_pts : int
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if num_pts % 2 != 0:
        num_pts += 1

    # Rayleigh range
    zr = length / np.sqrt((w1/w0)**2 - 1)

    x = np.linspace(0, length, num_pts)
    y = w0 * np.sqrt(1 + (x/zr)**2) / 2

    vertices = _mirror(x=x, y=y, axis='y')
    
    return vertices

def euler_taper(w0:float, w1:float, theta_max:float, rad2dy:float, length=None, alpha=0.5, num_pts=100) -> np.ndarray[float]:
    """
    Generates vertices for Euler taper in clockwise order.

    Parameters
    ----------
    w0 : float
        Width of the narrow end of the taper. Must be less than w1.
    w1 : float
        Width of the wide end of the taper. Must be greater than w0.
    theta_max : float
        Maximum taper half-angle in degrees. This angle occurs at the turning point.
    rad2dy : float
        Ratio between minimum bend radius to vertical displacement.
    length : int or float, optional
        Distance between both ends of the taper.
    alpha : float, optional
        Normalized position along taper length where the minimum bend radius occurs. Must be a value between 0 and 1.
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `w0` is less than `w1`.
    ValueError
        `alpha` is outside the range (0,1).
    """
    if w0 > w1:
        raise ValueError('Input parameter <w0> must be larger than <w1>.')
    if alpha <= 0 or alpha >= 1:
        raise ValueError('Input parameter <alpha> must be between 0 and 1.')
    if num_pts % 2 != 0:
        num_pts += 1
    
    # First part of curve before turning point
    span1 = abs(w0/2 - w1/2)
    min_radius1 = alpha * span1 * rad2dy
    x1, y1, _ = _euler_curve(
        min_radius=min_radius1, 
        angle_range=theta_max, 
        num_pts=int(num_pts/2))
    
    # Last part of curve after turning point
    span2 = abs(w0 - w1) - span1
    min_radius2 = (1-alpha) * span2 * rad2dy
    x2, y2, _ = _euler_curve(
        min_radius=min_radius2, 
        angle_range=theta_max,
        num_pts=int(num_pts/2))
    
    length_total = abs(x1[0] - x1[-1]) + abs(x2[0] - x2[-1])
    span_total = abs(y1[0] - y1[-1]) + abs(y2[0] - y2[-1])

    x2, y2 = -x2[::-1], -y2[::-1]
    x2 = x2 + length_total
    y2 = y2 + span_total

    # Combine first and last parts of the top half of taper
    x3 = np.hstack((x1,x2))
    y3 = np.hstack((y1,y2))

    dx = x3[-1]
    dy = y3[-1]

    # If <length> is given, scale along x axis
    if isinstance(length, (int, float)):
        x_error = length / dx
        x3 *= x_error

    # Since rad2dy is calculated via curve-fitting, correct for the error in the y coordinates by scaling.
    y_error = span1 / dy
    y3 *= y_error
    y3 += w0/2

    # Bottom half of taper by duplicating and flipping top half
    x4 = x3[::-1]
    y4 = -y3[::-1]

    # Combine top and bottom halves
    xt = np.hstack((x3,x4))
    yt = np.hstack((y3,y4))

    vertices = np.vstack((xt,yt)).T

    return vertices