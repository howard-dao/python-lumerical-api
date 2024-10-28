"""
Python module for generating directional couplers.
"""

import numpy as np
from geometry import _translate, _reflect, _stitch
from bends import circular_s_bend

def circular_directional_coupler(width:float, radius:float, span:float, length:float, gap:float, num_pts=100):
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
    length : float
        Straight path length.
    gap : float
        Separation between paths.
    num_pts : int, optional
        Number of vertices for one S-bend.

    Returns
    ----------
    v1 : ndarray
        Input-to-through vertices.
    v2 : ndarray
        Add-to-cross vertices.
    """
    if length < 0:
        raise ValueError(f'Input parameter <length> must be at least 0. It was given {length}.')
    sbend1 = circular_s_bend(
        width=width,
        radius=radius,
        span=span,
        reflect=True,
        num_pts=int(num_pts/2))
    
    L1 = max(sbend1[:,0]) - min(sbend1[:,0])

    str_wg = np.array([
        [L1, -span-width/2],
        [L1+length, -span-width/2],
        [L1+length, -span+width/2],
        [L1, -span+width/2]])
    
    sbend2 = _reflect(vertices=sbend1, angle=0)
    sbend2 = _translate(vertices=sbend2, dx=L1+length, dy=-span)
    sbend2 = np.flipud(sbend2)

    verts1 = _stitch(sbend1, str_wg, sbend2)
    verts2 = _reflect(vertices=verts1, angle=0)
    verts2 = _translate(vertices=verts2, dy=-2*span-gap)

    return verts1, verts2