"""
Python module for generating crosses
"""

import numpy as np
from geometry import _translate, _rotate, _stitch
from tapers import *

def simple_linear_cross(w0:float, w1:float, taper_length:float, straight_length:float=0.0):
    """
    Generates a simple cross with linear tapers.

    Parameters
    ----------
    w0 : float
        Input width.
    w1 : float
        Cross center width.
    taper_length : float
        Distance between both ends of the taper.
    straight_length : float, optional
        Length of straight path between taper and center.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `w1` is less than `w0`.
    """
    if w1 < w0:
        raise ValueError('Input parameter <w1> must be greater than or equal to <w0>.')
    
    # Create taper
    v1 = linear_taper(
        w0=w0,
        w1=w1,
        length=taper_length)
    v1 = _translate(vertices=v1, dx=-taper_length-straight_length-w1/2)

    # Create straight path between taper and center.
    s1 = np.array([
        [-straight_length-w1/2, w1/2],
        [-w1/2, w1/2],
        [-w1/2, -w1/2],
        [-straight_length-w1/2, -w1/2]])
    v1 = _stitch(v1, s1)
    
    # Create the three remaining ports
    v2 = _rotate(vertices=v1, angle=90)
    v3 = _rotate(vertices=v1, angle=180)
    v4 = _rotate(vertices=v1, angle=270)

    # Roll elements for stitching.
    v2 = np.roll(v2, shift=int(len(v2)/2), axis=0)
    v3 = np.roll(v3, shift=int(len(v3)/2), axis=0)
    v4 = np.roll(v4, shift=int(len(v4)/2), axis=0)

    # Stitch all ports together
    vertices = _stitch(v1, v2)
    vertices = _stitch(vertices, v3, fraction=1/4)
    vertices = _stitch(vertices, v4, fraction=1/6)

    return vertices

def simple_parabolic_cross(w0:float, w1:float, taper_length:float, straight_length:float=0.0, num_pts:int=100):
    """
    Generates a simple cross with parabolic tapers.

    Parameters
    ----------
    w0 : float
        Input width.
    w1 : float
        Cross center width.
    taper_length : float
        Distance between both ends of the taper.
    straight_length : float, optional
        Length of straight path between taper and center.
    num_pts : int, optional
        Number of vertices on one side of each taper.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `w1` is less than `w0`.
    """
    if w1 < w0:
        raise ValueError('Input parameter <w1> must be greater than or equal to <w0>.')
    
    # Create taper
    v1 = parabolic_taper(
        w0=w0,
        w1=w1,
        length=taper_length,
        num_pts=num_pts)
    v1 = _translate(vertices=v1, dx=-taper_length-straight_length-w1/2)

    # Create straight path between taper and center.
    s1 = np.array([
        [-straight_length-w1/2, w1/2],
        [-w1/2, w1/2],
        [-w1/2, -w1/2],
        [-straight_length-w1/2, -w1/2]])
    v1 = _stitch(v1, s1)
    
    # Create the three remaining ports
    v2 = _rotate(vertices=v1, angle=90)
    v3 = _rotate(vertices=v1, angle=180)
    v4 = _rotate(vertices=v1, angle=270)

    # Roll elements for stitching.
    v2 = np.roll(v2, shift=int(len(v2)/2), axis=0)
    v3 = np.roll(v3, shift=int(len(v3)/2), axis=0)
    v4 = np.roll(v4, shift=int(len(v4)/2), axis=0)

    # Stitch all ports together
    vertices = _stitch(v1, v2)
    vertices = _stitch(vertices, v3, fraction=1/4)
    vertices = _stitch(vertices, v4, fraction=1/6)

    return vertices

def simple_gaussian_cross(w0:float, w1:float, taper_length:float, straight_length:float=0.0, num_pts:int=100):
    """
    Generates a simple cross with gaussian tapers.

    Parameters
    ----------
    w0 : float
        Input width.
    w1 : float
        Cross center width.
    taper_length : float
        Distance between both ends of the taper.
    straight_length : float, optional
        Length of straight path between taper and center.
    num_pts : int, optional
        Number of vertices on one side of each taper.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `w1` is less than `w0`.
    """
    if w1 < w0:
        raise ValueError('Input parameter <w1> must be greater than or equal to <w0>.')
    
    # Create taper
    v1 = gaussian_taper(
        w0=w0,
        w1=w1,
        length=taper_length,
        num_pts=num_pts)
    v1 = _translate(vertices=v1, dx=-taper_length-straight_length-w1/2)

    # Create straight path between taper and center.
    s1 = np.array([
        [-straight_length-w1/2, w1/2],
        [-w1/2, w1/2],
        [-w1/2, -w1/2],
        [-straight_length-w1/2, -w1/2]])
    v1 = _stitch(v1, s1)
    
    # Create the three remaining ports
    v2 = _rotate(vertices=v1, angle=90)
    v3 = _rotate(vertices=v1, angle=180)
    v4 = _rotate(vertices=v1, angle=270)

    # Roll elements for stitching.
    v2 = np.roll(v2, shift=int(len(v2)/2), axis=0)
    v3 = np.roll(v3, shift=int(len(v3)/2), axis=0)
    v4 = np.roll(v4, shift=int(len(v4)/2), axis=0)

    # Stitch all ports together
    vertices = _stitch(v1, v2)
    vertices = _stitch(vertices, v3, fraction=1/4)
    vertices = _stitch(vertices, v4, fraction=1/6)

    return vertices