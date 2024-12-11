"""
Python module for generating rings.
"""

import numpy as np
from geometry import _translate, _rotate, _stitch
from bends import circular_arc, circular_u_bend, euler_u_bend

def circular_ring(width:float, radius:float, num_pts=100) -> np.ndarray[float]:
    """
    Generates a circular ring.
    
    Parameters
    ----------
    width : float
        Path width.
    radius : float
        Path center radius.
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    vertices = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=360, 
        num_pts=num_pts)
    return vertices

def circular_racetrack(width:float, radius:float, length:float, num_pts=100) -> np.ndarray[float]:
    """
    Generates a circular racetrack ring.

    Parameters
    ----------
    width : float
        Path width.
    radius : float
        Path center radius.
    length : float
        Straight track length.
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if length <= 0:
        raise ValueError('Input parameter <length> must be greater than 0.')
    
    ubend1 = circular_u_bend(
        width=width,
        span=2*radius,
        direction='counterclockwise',
        num_pts=int(num_pts/2))
    
    ubend2 = ubend1

    ubend1 = _translate(vertices=ubend1, dx=length/2)
    ubend2 = _rotate(vertices=ubend2, angle=180)
    ubend2 = _translate(vertices=ubend2, dx=-length/2, dy=2*radius)
    
    str_wg1 = np.array([
        [-length/2, width/2],
        [length/2, width/2],
        [length/2, -width/2],
        [-length/2, -width/2]])
    str_wg2 = np.array([
        [length/2, -width/2+2*radius],
        [-length/2, -width/2+2*radius],
        [-length/2, width/2+2*radius],
        [length/2, width/2+2*radius]])
    
    vertices = _stitch(str_wg1, ubend1, str_wg2, ubend2)

    return vertices

def euler_racetrack(width:float, span:float, length:float, num_pts=100) -> np.ndarray[float]:
    """
    Generates an Euler racetrack ring.

    Parameters
    ----------
    width : float
        Path width.
    span : float
        Separation between straight tracks.
    length : float
        Straight track length.
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if length <= 0:
        raise ValueError('Input parameter <length> must be greater than 0.')
    
    ubend1 = euler_u_bend(
        width=width,
        span=span,
        direction='counterclockwise',
        num_pts=int(num_pts/2))
    
    ubend2 = ubend1

    ubend1 = _translate(vertices=ubend1, dx=length/2)
    ubend2 = _rotate(vertices=ubend2, angle=180)
    ubend2 = _translate(vertices=ubend2, dx=-length/2, dy=span)

    str_wg1 = np.array([
        [-length/2, width/2],
        [length/2, width/2],
        [length/2, -width/2],
        [-length/2, -width/2]])
    str_wg2 = np.array([
        [length/2, -width/2+span],
        [-length/2, -width/2+span],
        [-length/2, width/2+span],
        [length/2, width/2+span]])
    
    vertices = _stitch(str_wg1, ubend1, str_wg2, ubend2)

    return vertices