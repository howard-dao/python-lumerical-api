"""
Python module for generating directional couplers.
"""

import numpy as np
from geometry import _translate, _reflect, _stitch
from bends import circular_s_bend, euler_s_bend

def circular_directional_coupler(width:float, radius:float, span:float, coupling_length:float, gap:float, num_pts=100) -> tuple[np.ndarray[float], np.ndarray[float], float]:
    """
    Generates a symmetric directional coupler with circular bends.

    Parameters
    ----------
    width : float
        Path width.
    radius : float
        Radius of curvature.
    span : float
        S-bend transverse length.
    coupling_length : float
        Length of the coupling region.
    gap : float
        Separation between coupling paths.
    num_pts : int, optional
        Number of vertices for one S-bend.

    Returns
    ----------
    verts1 : ndarray
        Input-to-through vertices.
    verts2 : ndarray
        Add-to-cross vertices.
    bend_length : float
        S-bend longitudinal length.

    Raises
    ----------
    ValueError
        `coupling_length` is negative.
    """
    if coupling_length < 0:
        raise ValueError(f'Input parameter <coupling_length> must be at least 0. It was given {coupling_length}.')
        
    sbend1 = circular_s_bend(
        width=width,
        radius=radius,
        span=span,
        reflect=True,
        num_pts=int(num_pts/2))
    
    bend_length = max(sbend1[:,0]) - min(sbend1[:,0])

    str_wg = np.array([
        [bend_length, -span-width/2],
        [bend_length+coupling_length, -span-width/2],
        [bend_length+coupling_length, -span+width/2],
        [bend_length, -span+width/2]])
    
    sbend2 = _reflect(vertices=sbend1, angle=0)
    sbend2 = _translate(vertices=sbend2, dx=bend_length+coupling_length, dy=-span)
    sbend2 = np.flipud(sbend2)

    verts1 = _stitch(sbend1, str_wg, sbend2)
    verts2 = _reflect(vertices=verts1, angle=0)
    verts2 = _translate(vertices=verts2, dy=-2*span-gap)

    return verts1, verts2, bend_length

def euler_directional_coupler(width:float, span:float, theta_max:float, coupling_length:float, gap:float, num_pts=100) -> np.ndarray[float]:
    """
    Generates a symmetric directional coupler with Euler bends.

    Parameters
    ----------
    width : float
        Path width.
    span : float
        S-bend transverse length.
    theta_max : float
        Maximum angle made by bend in degrees. Occurs at the turning point.
    coupling_length : float
        Length of the coupling region.
    gap : float
        Separation between coupling paths.
    num_pts : int, optional
        Number of vertices for one S-bend.

    Returns
    ----------
    verts1 : ndarray
        Input-to-through vertices.
    verts2 : ndarray
        Add-to-cross vertices.
    bend_length : float
        S-bend longitudinal length.

    Raises
    ----------
    ValueError
        `coupling_length` is negative.
    """
    if coupling_length < 0:
        raise ValueError(f'Input parameter <coupling_length> must be at least 0. It was given {coupling_length}.')
    
    sbend1 = euler_s_bend(
        width=width,
        rad2dy=1.0,
        theta_max=theta_max,
        span=span,
        reflect=True,
        num_pts=num_pts)
    
    bend_length = max(sbend1[:,0]) - min(sbend1[:,0])

    str_wg = np.array([
        [bend_length, -span-width/2],
        [bend_length+coupling_length, -span-width/2],
        [bend_length+coupling_length, -span+width/2],
        [bend_length, -span+width/2]])
    
    sbend2 = _reflect(vertices=sbend1, angle=0)
    sbend2 = _translate(vertices=sbend2, dx=bend_length+coupling_length, dy=-span)
    sbend2 = np.flipud(sbend2)

    verts1 = _stitch(sbend1, str_wg, sbend2)
    verts2 = _reflect(vertices=verts1, angle=0)
    verts2 = _translate(vertices=verts2, dy=-2*span-gap)

    return verts1, verts2, bend_length