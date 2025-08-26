"""
Python module for generating bends.
"""

import numpy as np
from geometry import _translate, _reflect, _rotate, _thicken, _circular_curve, _euler_curve

def circular_arc(radius:float, angle_range:float, angle_start=0.0, width=0.0, direction='counterclockwise', num_pts=100):
    """
    Generates a circular arc path.
    
    Parameters
    ----------
    radius : float
        Center radius of curvature.
    angle_range : float
        Angular range in degrees.
    angle_start : float, optional
        Initial angle in degrees. Zero degrees points to the +x direction.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    direction : str, optional
        Either "clockwise" or "counterclockwise".
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `width` is larger than the diameter, or `2*radius`.
    ValueError
        `angle_range` is zero, negative, or greater than 360.
    ValueError
        `direction` is neither 'clockwise' or 'counterclockwise'.
    """
    # Check parameters
    if width < 0.0 or width >= 2*radius:
        raise ValueError(
            'Input parameter <width> must be in range [0, 2*<radius>).')
    if angle_range <= 0 or angle_range > 360:
        raise ValueError(
            'Input parameter <angle_range> must be between 0 and 360.')
    if angle_start >= 360:
        angle_start = angle_start % 360
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    vertices, theta = _circular_curve(
        radius=radius,
        angle_range=angle_range,
        angle_start=angle_start,
        direction=direction,
        num_pts=num_pts)
    
    # Add width to curve
    if width > 0:
        vertices = _thicken(
            curve=vertices,
            width=width,
            angle_start=angle_start,
            angle_final=angle_start+angle_range)

    return vertices

def circular_s_bend(radius:float, width=0.0, span:float=None, angle_range:float=None, reflect=False, num_pts=100):
    """
    Generates a circular S-bend.

    Parameters
    ----------
    radius : float
        Path center radius of curvature.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    span : float
        Longitudinal distance between input and output.
    angle_range : float, optional
        Angular range of bend until turning point.
    reflect : bool, optional
        Whether to reflect over longitudinal axis.
    num_pts : int, optional
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `span` is larger than `2*radius`.
    """
    if span != None:
        if span > 2*radius:
            raise ValueError('Input parameter <span> must be less than 2*<radius>.')
        theta0 = np.rad2deg(np.arccos(np.sqrt((radius*span - span**2/4)/radius**2)))
        delta_theta = 90 - theta0
    elif angle_range != None:
        if angle_range <= 0 or angle_range > 90:
            raise ValueError('Input parameter <angle_range> must be between 0 and 180 degrees.')
        span = 2 * radius * (1 - np.sin(np.deg2rad(angle_range)))
        delta_theta = angle_range
    else:
        raise ValueError('Input parameters <span> and <angle_range> are not given.')
    
    # Create curve
    curve1, _ = _circular_curve(
        radius=radius,
        angle_range=delta_theta,
        angle_start=0,
        direction='counterclockwise',
        num_pts=int(num_pts/2))
    curve2, _ = _circular_curve(
        radius=radius,
        angle_range=delta_theta,
        angle_start=delta_theta,
        direction='clockwise',
        num_pts=int(num_pts/2))

    x1, y1 = curve1[:,0], curve1[:,1]
    x2, y2 = curve2[1:,0], curve2[1:,1]
    
    dx = max(x1) - min(x1)
    dy = max(y1) - min(y1)

    x2 += dx
    y2 += dy
    
    xt = np.hstack((x1, x2))
    yt = np.hstack((y1, y2))
    vertices = np.vstack((xt, yt)).T

    # Flip upside down if true
    if reflect:
        vertices = _reflect(vertices, angle=0)

    # Add width to curve
    if width > 0:
        vertices = _thicken(
            curve=vertices,
            width=width,
            angle_start=0,
            angle_final=0)

    return vertices

def circular_u_bend(span:float, width=0.0, direction='counterclockwise', num_pts=100):
    """
    Generates a circular 180 degree bend path.

    Parameters
    ----------
    span : float
        Distance between input and output.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    direction : str, optional
        Either "clockwise" or "counterclockwise".
    num_pts : int
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    vertices = circular_arc(
        width=width,
        radius=span/2,
        angle_range=180,
        angle_start=0,
        direction=direction,
        num_pts=num_pts)
    return vertices

def circular_l_bend(radius:float, width=0.0, direction='counterclockwise', num_pts=100):
    """
    Generates a circular 90 degree bend path.

    Parameters
    ----------
    radius : float
        Bend radius.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    direction : str
        Either "clockwise" or "counterclockwise".
    num_pts : int
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    vertices = circular_arc(
        width=width,
        radius=radius,
        angle_range=90,
        angle_start=0,
        direction=direction,
        num_pts=num_pts)
    return vertices

def euler_arc(width:float, min_radius:float, angle_range:float, angle_start=0.0, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler (aka clothoidal) arc path.

    Parameters
    ----------
    width : float
        Path width.
    min_radius : float
        Minimum radius of curvature.
    angle_range : float
        Arc angular range in degrees.
    angle_start : float, optional
        Arc start angle in degrees. Zero degrees points to the +x direction.
    direction : str, optional
        Either "clockwise" or "counterclockwise".
    num_pts : int, optional
        Number of points on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `width` is less than `2*min_radius`.
    ValueError
        `direction` is neither 'clockwise' or 'counterclockwise'.
    """
    if width >= 2*min_radius:
        raise ValueError('Input parameter <width> must be less than 2*<min_radius>.')
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError('Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    curve, theta = _euler_curve(
        min_radius=min_radius,
        angle_range=angle_range, 
        angle_start=angle_start,
        num_pts=num_pts)
    
    # Add width to curve
    vertices = _thicken(
        curve=curve,
        width=width,
        angle_start=angle_start,
        angle_final=angle_start+angle_range)

    if direction == 'clockwise':
        vertices = _reflect(vertices=vertices, angle=angle_start)

    return vertices

def euler_s_bend(rad2dy:float, theta_max:float, span:float, width=0.0, length:int|float=None, reflect=False, num_pts=100):
    """
    Generates an Euler S-bend.

    Parameters
    ----------
    rad2dy : float
        Ratio between minimum bend radius to vertical displacement.
    theta_max : float
        Maximum angle made by bend in degrees. Occurs at the turning point.
    span : float
        Transverse distance between input and output.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    length : int or float, optional
        Longitudinal distance between input and output. 
    reflect : bool, optional
        Whether to reflect over longitudinal axis.
    num_pts : int, optional
        Number of points on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if theta_max <= 0 or theta_max > 90:
        raise ValueError(f'Input parameter <theta_max> must be between 0 and 90 degrees. It was given {theta_max} degrees.')

    # Generate curve
    min_radius = rad2dy * span/2

    # First part of curve before turning point
    curve1, theta1 = _euler_curve(
        min_radius=min_radius,
        angle_range=theta_max,
        num_pts=int(num_pts/2))
    x1, y1 = curve1[:,0], curve1[:,1]

    # Last part of curve after turning point
    x2, y2, theta2 = -x1[-2::-1], -y1[-2::-1], theta1[-2::-1]

    dx = 2 * x1[-1]
    dy = 2 * y1[-1]

    x2 = x2 + dx
    y2 = y2 + dy

    # Combine first and last parts
    xt = np.hstack((x1, x2))
    yt = np.hstack((y1, y2))
    theta = np.hstack((theta1, theta2))

    # If <length> is given, scale along x axis
    if isinstance(length, (int,float)):
        x_error = length / dx
        xt *= x_error

    # Since rad2dy is calculated via curve-fitting, correct for the error in the y coordinates by scaling.
    y_error = span / yt[-1]
    yt *= y_error

    vertices = np.vstack((xt,yt)).T

    # Flip upside down if true
    if reflect:
        vertices = _reflect(vertices=vertices, angle=0)

    # Add width to curve
    if width > 0.0:
        vertices = _thicken(
            curve=vertices,
            width=width,
            angle_start=0,
            angle_final=0)

    return vertices

def euler_u_bend(span:float, width=0.0, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler 180 degree bend path.

    Parameters
    ----------
    span : float
        Distance between input and output.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    direction : str, optional
        Either "clockwise" or "counterclockwise".
    num_pts : float
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `direction` is neither 'clockwise' or 'counterclockwise'.
    """
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')
    if num_pts % 2 != 0:
        num_pts += 1

    # Simulated relationship between minimum bend radius and span.
    rad_to_span_ratio = 0.7263051699691657
    min_radius = rad_to_span_ratio * span/2

    half_num_pts = int(num_pts/2)

    verts1, _ = _euler_curve(
        min_radius=min_radius,
        angle_range=90,
        angle_start=0,
        num_pts=half_num_pts)
    
    verts2 = _reflect(vertices=verts1, angle=0)
    verts2 = _translate(vertices=verts2, dy=span)
    
    vertices = np.vstack((verts1, verts2[-2::-1]))

    # Add width to curve
    if width > 0:
        vertices = _thicken(
            curve=vertices,
            width=width,
            angle_start=0,
            angle_final=180)

    return vertices

def euler_l_bend(min_radius:float, width=0.0, span=None, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler 90 degree bend.

    Parameters
    ----------
    min_radius : float
        Minimum radius of curvature.
    width : float, optional
        Path width. If given a value of zero, returns the vertices of a curve.
    direction : str, optional
        Either "clockwise" or "counterclockwise".
    num_pts : int
        Number of vertices on one side of the shape.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    ValueError
        `direction` is neither "clockwise" or "counterclockwise".
    """
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError('Input parameter <direction> must be either "clockwise" or "counterclockwise".')
    if num_pts % 2 != 0:
        num_pts += 1

    # First half
    curve1, theta1 = _euler_curve(
        min_radius=min_radius, 
        angle_range=45, 
        num_pts=int(num_pts/2))
    
    dx = max(curve1[:,0]) - min(curve1[:,0])
    dy = max(curve1[:,1]) - min(curve1[:,1])

    # Second half
    curve2 = _reflect(
        vertices=curve1[-2::-1],
        angle=0)
    curve2 = _rotate(
        vertices=curve2,
        angle=-90)
    curve2 = _translate(
        vertices=curve2,
        dx=dx+dy,
        dy=dx+dy)
    
    # Combine both halves
    curve = np.vstack((curve1, curve2))
    xt, yt = curve[:,0], curve[:,1]

    # Calculate angles
    theta2 = 90 - theta1[::-1]
    theta = np.hstack((theta1, theta2))
    
    # Resize if <span> is given
    if isinstance(span, (int, float)):
        x_error = span / xt[-1]
        y_error = span / yt[-1]
        xt *= x_error
        yt *= y_error

    vertices = np.vstack((xt,yt)).T
    
    # Add width to curve
    if width > 0:
        vertices = _thicken(
            curve=vertices,
            width=width,
            angle_start=0,
            angle_final=90)
    
    return vertices