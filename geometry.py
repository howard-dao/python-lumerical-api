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
    new_points = np.transpose(np.matmul(matrix, points.T))

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
    
    new_points = np.transpose(np.matmul(matrix, new_points.T))
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

    points = np.vstack((x,y)).T

    return points

def parabolic_taper(w0:float, w1:float, length:float, num_pts=100):
    """
    Generates vertices for parabolic taper in clockwise order.

    Parameters:
        w0 : float
            Width of one end of the taper.
        w1 : float
            Width of the other end of the taper.
        length : float
            Distance between both ends of the taper.
        num_pts : int
            Number of points with which to draw one side of the taper.

    Returns:
        points : [N-by-2] ndarray
            Vertices in clockwise order starting from the top left vertex.
    """
    a = 4 * length / (w1**2 - w0**2)
    c = a * w0**2 / 4

    x = np.linspace(0, length, num_pts)
    y = np.sqrt((x + c) / a)

    # Concatenate top and bottom points
    points_top = [(xp, yp) for xp, yp in zip(x, y)]
    points_bot = [(xp, -yp) for xp, yp in zip(x, y)]
    points_bot = list(reversed(points_bot))
    points = np.array(points_top + points_bot)
    
    return points

def gaussian_taper(w0:float, w1:float, length:float, num_pts=100):
    """
    Generates vertices for Gaussian taper in clockwise order.

    Parameters:
        w0 : float
            Width of one end of the taper.
        w1 : float
            Width of the other end of the taper.
        length : float
            Distance between both ends of the taper.
        num_pts : int
            Number of points with which to draw one side of the taper.

    Returns:
        points : [N-by-2] ndarray
            Vertices in clockwise order starting from the top left vertex.
    """
    zr = length / np.sqrt((w1/w0)**2 - 1)

    x = np.linspace(0, length, num_pts)
    y = w0 * np.sqrt(1 + (x/zr)**2) / 2

    points_top = [(xp, yp) for xp, yp in zip(x, y)]
    points_bot = [(xp, -yp) for xp, yp in zip(x, y)]
    points_bot = list(reversed(points_bot))
    points = np.array(points_top + points_bot)
    
    return points

def euler_taper(w0:float, w1:float, theta_max:float, rad2dy:float, alpha:float, num_pts=100):
    """
    Generates vertices for Euler taper in clockwise order.

    Parameters:
        w0 : float
            Width of one end of the taper.
        w1 : float
            Width of the other end of the taper.
        theta_max : float
        rad2dy : float
        alpha : float
        num_pts : int
            Number of points with which to draw one side of the taper.
    """
    span1 = abs(w0/2 - w1/2)
    min_radius1 = alpha * span1 * rad2dy
    x1, y1, theta1 = _euler_curve(
        min_radius=min_radius1, 
        angle_range=theta_max, 
        num_pts=num_pts)
    
    span2 = abs(w0 - w1) - span1
    min_radius2 = (1-alpha) * span2 * rad2dy
    x2, y2, theta2 = _euler_curve(
        min_radius=min_radius2, 
        angle_range=theta_max,
        num_pts=num_pts)
    
    length_total = abs(x1[0] - x1[-1]) + abs(x2[0] - x2[-1])
    span_total = abs(y1[0] - y1[-1]) + abs(y2[0] - y2[-1])

    x2, y2, theta2 = -x2[::-1], -y2[::-1], theta2[::-1]

    x2 = x2 + length_total
    y2 = y2 + span_total

    x3      = np.hstack((x1, x2))
    y3      = np.hstack((y1, y2)) + w0/2
    # theta3  = np.hstack((theta1, theta2))

    x4      = x3[::-1]
    y4      = -y3[::-1]
    # theta4  = theta4[::-1]

    xt      = np.hstack((x3, x4))
    yt      = np.hstack((y3, y4))
    # thetat  = np.hstack((theta3, theta4))

    points = np.vstack((xt, yt)).T

    return points

def circular_arc(width:float, radius:float, angle_range:float, angle_start=0, direction='counterclockwise',  num_pts=100):
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
        num_pts : int, optional
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

    theta = np.linspace(angle_start, angle_start+angle_range, round(num_pts/2))
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

def circular_ring(width:float, radius:float, num_pts=100):
    """
    Generates a circular ring.
    
    Parameters:
        width : float
            Arc width.
        radius : float
            Arc center radius.
        num_pts : int, optional
            Number of points.

    Returns:
        [N-by-2] ndarray : Ring vertices.
    """
    points = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=360, 
        num_pts=num_pts)
    return points

def circular_s_bend(width:float, radius:float, span:float, reflect=False, angle_start=0, num_pts=100):
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
        angle_start : float, optional
            Longitudinal angular direction in degrees.
        num_pts : int, optional
            Number of points.
    """
    if num_pts % 2 != 0:
        num_pts += 1

    theta1 = np.rad2deg(np.arccos(np.sqrt((radius*span - span**2/4)/radius**2)))
    angle_range = 90 - theta1

    points1 = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=angle_range, 
        angle_start=0, 
        direction='clockwise', 
        num_pts=num_pts)
    points2 = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=angle_range, 
        angle_start=-angle_range, 
        direction='counterclockwise', 
        num_pts=num_pts)
    
    # Align the second arc with the first arc.
    theta2 = np.deg2rad(90 - angle_range)
    length = 2 * radius * np.sqrt(1 - np.sin(theta2)**2)
    points2 = _translate(points2, dx=length/2, dy=-span/2)

    idx = int(num_pts/2)
    points = np.vstack((points1[:idx], points2[::-1], points1[idx:]))

    if reflect:
        points = _reflect(points, angle=0)

    if angle_start != 0:
        points = _rotate(points, angle=angle_start)

    return points

def _clothoid_ode_rhs(state, t, kappa0, kappa1):
    """
    Helper function to set up Euler ODEs.
    """
    x, y, theta = state[0], state[1], state[2]
    return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1*t])

def _euler_curve(min_radius:float, angle_range:float, angle_start=0, num_pts=100):
    """
    Helper function to create an Euler curve.
    """
    x0, y0, theta0 = 0, 0, np.deg2rad(angle_start)
    thetas = np.deg2rad(angle_range)
    L = 2 * min_radius * thetas
    t = np.linspace(0, L, num_pts)
    kappa0 = 0
    kappa1 = 2 * thetas / L**2

    sol = odeint(
        func=_clothoid_ode_rhs, 
        y0=np.array([x0,y0,theta0]), 
        t=t, 
        args=(kappa0,kappa1))

    x, y, theta = sol[:,0], sol[:,1], sol[:,2]

    return x, y, theta

def euler_arc(width:float, min_radius:float, angle_range:float, angle_start=0, direction='counterclockwise', num_pts=100):
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
        num_pts : int, optional
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
        num_pts=num_pts)

    x_inner = xt + (width/2)*np.cos(theta + np.pi/2)
    y_inner = yt + (width/2)*np.sin(theta + np.pi/2)

    x_outer = xt + (width/2)*np.cos(theta - np.pi/2)
    y_outer = yt + (width/2)*np.sin(theta - np.pi/2)

    x = np.hstack((x_inner, x_outer[::-1]))
    y = np.hstack((y_inner, y_outer[::-1]))
    
    # points = np.array([(xp, yp) for xp, yp in zip(x, y)])
    points = np.vstack((x,y)).T

    if direction == 'clockwise':
        points = _reflect(points=points, angle=angle_start)

    return points

def euler_u_bend(width:float, span:float, angle=0, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler 180 degree bend path.
    """
    if num_pts % 2 != 0:
        num_pts += 1
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    # Simulated relationship between minimum bend radius and span.
    rad_to_span_ratio = 0.7263051699691657
    min_radius = rad_to_span_ratio * span/2

    half_n_points = round(num_pts/2)

    points1 = euler_arc(
        width=width, 
        min_radius=min_radius, 
        angle_range=90, 
        angle_start=0, 
        direction=direction, 
        num_pts=half_n_points)
    
    points2 = _reflect(points1, angle=0)
    points2 = _translate(points=points2, dy=span)
    
    points = np.vstack((
        points1[:half_n_points-1], 
        points2[half_n_points-1::-1], 
        points2[-1:half_n_points:-1], 
        points1[half_n_points:]))

    return points

def euler_l_bend(width:float, min_radius:float, direction='counterclockwise', num_pts=100):
    if num_pts % 2 != 0:
        num_pts += 1
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')
    
    # rad_to_span_ratio = 2.5415151437386307  / 2
    # min_radius = rad_to_span_ratio * span

    half_n_points = round(num_pts/2)

    xt, yt, theta = _euler_curve(
        min_radius=min_radius, 
        angle_range=45, 
        num_pts=half_n_points)
    
    x_inner = xt + (width/2)*np.cos(theta + np.pi/2)
    y_inner = yt + (width/2)*np.sin(theta + np.pi/2)

    x_outer = xt + (width/2)*np.cos(theta - np.pi/2)
    y_outer = yt + (width/2)*np.sin(theta - np.pi/2)

    x = np.hstack((x_inner, x_outer[::-1]))
    y = np.hstack((y_inner, y_outer[::-1]))

    points1 = np.vstack((x,y)).T

    points2 = _reflect(points1, angle=0)
    points2 = _rotate(points2, angle=-90)
    points2 = _translate(points2, dx=max(xt)+max(yt), dy=max(xt)+max(yt))

    points = np.vstack((
        points1[:half_n_points-1], 
        points2[half_n_points-1::-1], 
        points2[-1:half_n_points:-1], 
        points1[half_n_points:]))
    
    return points