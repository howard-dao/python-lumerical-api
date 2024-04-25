"""
Python module containing various shapes for photonic paths.
Author(s): Howard Dao
"""

import numpy as np
from scipy.integrate import odeint
import copy

def _translate(points:np.ndarray, dx=0, dy=0):
    """
    Translates vertices along either x and y directions.

    Parameters:
        points : [N-by-2] ndarray
            Shape vertices.
        dx : float, optional
            Distance along x.
        dy : float, optional
            Distance along y.

    Returns:
        [N-by-2] array : Translated vertices.
    """
    new_points = copy.copy(points)
    new_points[:,0] += dx
    new_points[:,1] += dy

    return new_points

def _reflect(points:np.ndarray, angle:float):
    """
    Reflects vertices with respect to a given angle.

    Parameters:
        points : [N-by-2] ndarray
            Shape vertices.
        angle : float
            Angle over which to reflect vertices in degrees.

    Returns:
        [N-by-2] array : Reflected vertices.
    """
    angle = np.deg2rad(angle)
    matrix = np.array([[np.cos(2*angle), np.sin(2*angle)],
                       [np.sin(2*angle), -np.cos(2*angle)]])
    new_points = np.transpose(np.matmul(matrix, np.transpose(points)))

    return new_points

def _rotate(points:np.ndarray, angle:float, origin=[0,0]):
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters:
        points : [N-by-2] ndarray
            Shape vertices.
        angle : float
            Angle of rotation in degrees.
        origin : [1-by-2] array-like
            Point about which to rotate.

    Returns:
        [N-by-2] ndarray : Rotated vertices.
    """
    angle = np.deg2rad(angle)
    ox,oy = origin
    new_points = _translate(points, dx=-ox, dy=-oy)

    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    
    new_points = np.transpose(np.matmul(matrix, np.transpose(new_points)))
    new_points = _translate(new_points, dx=ox, dy=oy)

    return new_points

def linear_taper(w0:float, w1:float, length:float):
    """
    Generates vertices for linear taper in clockwise order.

    Parameters:
        w0 : float
            Width of one end of the taper.
        w1 : float
            Width of the other end of the taper.
        length : float
            Distance between both ends of the taper.

    Returns:
        points : [N-by-2] ndarray
            Vertices in clockwise order, starting from the top left vertex.
    """
    x = [0, length, length, 0]
    y = [w0/2, w1/2, -w1/2, -w0/2]

    points = np.transpose(np.vstack((x,y)))

    return points

def parabolic_taper(w0:float, w1:float, length:float, n_points=100):
    """
    Generates vertices for parabolic taper in clockwise order.

    Parameters:
        w0 : float
            Width of one end of the taper.
        w1 : float
            Width of the other end of the taper.
        length : float
            Distance between both ends of the taper.
        n_points : int
            Number of points with which to draw one side of the taper.

    Returns:
        points : [N-by-2] ndarray
            Vertices in clockwise order starting from the top left vertex.
    """
    a = 4 * length / (w1**2 - w0**2)
    c = a * w0**2 / 4

    x = np.linspace(0, length, n_points)
    y = np.sqrt((x + c) / a)

    # Concatenate top and bottom points
    points_top = [(xp, yp) for xp, yp in zip(x, y)]
    points_bot = [(xp, -yp) for xp, yp in zip(x, y)]
    points_bot = list(reversed(points_bot))
    points = np.array(points_top + points_bot)
    
    return points

def circular_arc(width:float, radius:float, angle_range:float, angle_start=0, direction='counterclockwise',  n_points=100):
    """
    Generates a circular arc path.
    
    Parameters:
        width : float
            Arc width.
        radius : float
            Arc center radius.
        angle_range : float
            Arc angular range in degrees.
        angle_start : float, optional
            Arc start angle in degrees. Zero degrees points to the +x direction.
        direction : str, optional
            Direction of the arc from starting point.
        n_points : int, optional
            Number of points.

    Returns:
        [N-by-2] ndarray : Arc vertices.

    Raises:
        ValueError: <width> is larger than the diameter.
        ValueError: <angle_range> is zero, negative, or greater than 360.
        ValueError: <direction> is neither 'clockwise' or 'counterclockwise'.
    """
    inner_radius = radius - width/2
    outer_radius = radius + width/2

    # Check parameters
    if inner_radius <= 0:
        raise ValueError(
            'Input parameter <width> must be less than 2*<radius>.')
    if angle_range <= 0 or angle_range > 360:
        raise ValueError(
            'Input parameter <angle_range> must be between 0 and 360.')
    if angle_start >= 360:
        angle_start = angle_start % 360
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    theta = np.linspace(angle_start, angle_start+angle_range, round(n_points/2))
    theta = np.deg2rad(theta-90)

    # Draw inner arc
    x_inner = inner_radius * np.cos(theta)
    y_inner = inner_radius * np.sin(theta)

    # Draw outer arc
    x_outer = outer_radius * np.cos(theta)
    y_outer = outer_radius * np.sin(theta)

    # Concatenate inner and outer points
    points_inner = [(xp, yp) for xp, yp in zip(x_inner, y_inner)]
    points_outer = [(xp, yp) for xp, yp in zip(x_outer, y_outer)]
    points_outer = list(reversed(points_outer))
    points = np.array(points_inner + points_outer)

    # Move points so that the start of the arc is at the origin points (0,0)
    dx = (points[0,0] + points[-1,0]) / 2
    dy = (points[0,1] + points[-1,1]) / 2
    points = _translate(points=points, dx=-dx, dy=-dy)

    if direction == 'clockwise':
        points = _reflect(points=points, angle=angle_start)

    return points

def circular_s_bend(width:float, radius:float, span:float, reflect=False, angle=0, n_points=100):
    """
    Generates a circular S-bend.

    Parameters:
        width : float
            Path width.
        radius : float
            Arc center radius of curvature.
        span : float
            Lateral distance.
        reflect : bool, optional
            Whether to reflect over longitudinal axis.
        angle : float, optional
            Longitudinal angular direction in degrees.
        n_points : int, optional
            Number of points.
    """
    if n_points % 2 != 0:
        n_points += 1

    theta1 = np.rad2deg(np.arccos(np.sqrt((radius*span - span**2/4)/radius**2)))
    angle_range = 90 - theta1

    points1 = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=angle_range, 
        angle_start=0, 
        direction='clockwise', 
        n_points=n_points)
    points2 = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=angle_range, 
        angle_start=-angle_range, 
        direction='counterclockwise', 
        n_points=n_points)
    
    # Align the second arc with the first arc.
    theta2 = np.deg2rad(90 - angle_range)
    length = 2 * radius * np.sqrt(1 - np.sin(theta2)**2)
    points2 = _translate(points2, dx=length/2, dy=-span/2)

    idx = int(n_points/2)
    points = np.vstack((points1[:idx], points2[::-1], points1[idx:]))

    if reflect:
        points = _reflect(points, angle=0)

    if angle != 0:
        points = _rotate(points, angle=angle)

    return points

def _clothoid_ode_rhs(state, t, kappa0, kappa1):
    """
    Helper function to set up Euler ODEs.
    """
    x, y, theta = state[0], state[1], state[2]
    return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1*t])

def _euler_curve(min_radius:float, angle_range:float, angle_start=0, n_points=100):
    """
    Helper function to create an Euler curve.
    """
    x0, y0, theta0 = 0, 0, np.deg2rad(angle_start)
    thetas = np.deg2rad(angle_range)
    L = 2 * min_radius * thetas
    t = np.linspace(0, L, n_points)
    kappa0 = 0
    kappa1 = 2 * thetas / L**2

    sol = odeint(
        func=_clothoid_ode_rhs, 
        y0=np.array([x0,y0,theta0]), 
        t=t, 
        args=(kappa0,kappa1))

    x, y, theta = sol[:,0], sol[:,1], sol[:,2]

    return x, y, theta

def euler_arc(width:float, min_radius:float, angle_range:float, angle_start=0, direction='counterclockwise', n_points=100):
    """
    Generates an Euler (aka clothoidal) arc path.

    Parameters:
        width : float
            Arc width.
        min_radius : float
            Minimum radius of curvature.
        angle_range : float
            Arc angular range in degrees.
        angle_start : float, optional
            Arc start angle in degrees. Zero degrees points to the +x direction.
        direction : str, optional
            Direction of the arc from starting point.
        n_points : int, optional
            Number of points.
    """
    if min_radius <= width/2:
        raise ValueError('Input parameter <width> must be less than 2*<min_radius>.')
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    xt, yt, theta = _euler_curve(
        min_radius=min_radius, 
        angle_range=angle_range, 
        angle_start=angle_start, 
        n_points=n_points)

    x_inner = xt + (width/2)*np.cos(theta + np.pi/2)
    y_inner = yt + (width/2)*np.sin(theta + np.pi/2)

    x_outer = xt + (width/2)*np.cos(theta - np.pi/2)
    y_outer = yt + (width/2)*np.sin(theta - np.pi/2)

    x = np.hstack((x_inner, x_outer[::-1]))
    y = np.hstack((y_inner, y_outer[::-1]))
    
    # points = np.array([(xp, yp) for xp, yp in zip(x, y)])
    points = np.transpose(np.vstack((x,y)))

    if direction == 'clockwise':
        points = _reflect(points=points, angle=angle_start)

    return points

def euler_u_bend(width:float, span:float, angle=0, direction='counterclockwise', n_points=100):
    """
    Generates an Euler 180 degree bend path.
    """
    if n_points % 2 != 0:
        n_points += 1
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    rad_to_span_ratio = 0.7263051699691657 / 2
    min_radius = rad_to_span_ratio * span

    half_n_points = round(n_points/2)

    points1 = euler_arc(
        width=width, 
        min_radius=min_radius, 
        angle_range=90, 
        angle_start=0, 
        direction=direction, 
        n_points=half_n_points)
    
    points2 = _reflect(points1, angle=0)
    points2 = _translate(points=points2, dy=span)
    
    points = np.vstack((
        points1[:half_n_points-1], 
        points2[half_n_points-1::-1], 
        points2[-1:half_n_points:-1], 
        points1[half_n_points:]))

    return points

def euler_l_bend(width:float, min_radius:float, direction='counterclockwise', n_points=100):
    if n_points % 2 != 0:
        n_points += 1
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')
    
    # rad_to_span_ratio = 2.5415151437386307  / 2
    # min_radius = rad_to_span_ratio * span

    half_n_points = round(n_points/2)

    xt, yt, theta = _euler_curve(
        min_radius=min_radius, 
        angle_range=45, 
        n_points=half_n_points)
    
    x_inner = xt + (width/2)*np.cos(theta + np.pi/2)
    y_inner = yt + (width/2)*np.sin(theta + np.pi/2)

    x_outer = xt + (width/2)*np.cos(theta - np.pi/2)
    y_outer = yt + (width/2)*np.sin(theta - np.pi/2)

    x = np.hstack((x_inner, x_outer[::-1]))
    y = np.hstack((y_inner, y_outer[::-1]))

    points1 = np.transpose(np.vstack((x,y)))

    points2 = _reflect(points1, angle=0)
    points2 = _rotate(points2, angle=-90)
    points2 = _translate(points2, dx=max(xt)+max(yt), dy=max(xt)+max(yt))

    points = np.vstack((
        points1[:half_n_points-1], 
        points2[half_n_points-1::-1], 
        points2[-1:half_n_points:-1], 
        points1[half_n_points:]))
    
    return points