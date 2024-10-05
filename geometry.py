"""
Python module containing various shapes for photonic paths.
Author(s): Howard Dao
"""

import numpy as np
from scipy.integrate import odeint
import copy

def _translate(vertices:np.ndarray, dx=0, dy=0):
    """
    Translates vertices along either x and y directions.

    Parameters:
        vertices : [N-by-2] ndarray
            Shape vertices.
        dx : float, optional
            Distance along x.
        dy : float, optional
            Distance along y.

    Returns:
        [N-by-2] array : Translated vertices.
    """
    new_vertices = copy.copy(vertices)
    new_vertices[:,0] += dx
    new_vertices[:,1] += dy

    return new_vertices

def _reflect(vertices:np.ndarray, angle:float):
    """
    Reflects vertices with respect to a given angle.

    Parameters:
        vertices : [N-by-2] ndarray
            Shape vertices.
        angle : float
            Angle over which to reflect vertices in degrees.

    Returns:
        [N-by-2] array : Reflected vertices.
    """
    angle = np.deg2rad(angle)
    matrix = np.array([[np.cos(2*angle), np.sin(2*angle)],
                       [np.sin(2*angle), -np.cos(2*angle)]])
    new_vertices = np.transpose(np.matmul(matrix, vertices.T))

    return new_vertices

def _rotate(vertices:np.ndarray, angle:float, origin=[0,0]):
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters:
        vertices : [N-by-2] ndarray
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
    new_vertices = _translate(vertices, dx=-ox, dy=-oy)

    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    
    new_vertices = np.transpose(np.matmul(matrix, new_vertices.T))
    new_vertices = _translate(new_vertices, dx=ox, dy=oy)

    return new_vertices

def _thicken(x:np.ndarray, y:np.ndarray, theta:np.ndarray, width:float):
    """
    Adds width to a curve.

    Parameters:
        x : ndarray
            x data.
        y : ndarray
            y data.
        theta : ndarray
            Angle of the line tangent to the curve, in degrees.
        width : float
            Path width.

    Returns:
        vertices : ndarray
            Shape vertices.
    """
    theta_rad = np.deg2rad(theta)

    x1 = x + (width/2)*np.cos(theta_rad + np.pi/2)
    y1 = y + (width/2)*np.sin(theta_rad + np.pi/2)

    x2 = x + (width/2)*np.cos(theta_rad - np.pi/2)
    y2 = y + (width/2)*np.sin(theta_rad - np.pi/2)

    x = np.hstack((x1, x2[::-1]))
    y = np.hstack((y1, y2[::-1]))

    vertices = np.vstack((x,y)).T

    return vertices

def linear_taper(w0:float, w1:float, length:float, num_pts=2):
    """
    Generates vertices for linear taper in clockwise order.

    Parameters:
        w0 : float
            Width of one end of the taper.
        w1 : float
            Width of the other end of the taper.
        length : float
            Distance between both ends of the taper.
        num_pts : int, optional
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.
    """
    if num_pts < 2:
        raise ValueError(f'Input parameter <num_pts> must be at least 2, but was given a value of {num_pts}.')
    if num_pts % 2 != 0:
        num_pts += 1

    slope = (w1-w0)/2 / length
    x = np.linspace(0, length, num_pts)
    y = slope*x + w0/2

    # Concatenate top and bottom vertices
    verts_top = np.vstack((x,y)).T
    verts_bot = np.vstack((x,-y)).T
    verts_bot = verts_bot[::-1]
    vertices = np.vstack((verts_top, verts_bot))

    return vertices

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
            Number of points on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.
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
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.
    """
    if num_pts % 2 != 0:
        num_pts += 1

    # Rayleigh range
    zr = length / np.sqrt((w1/w0)**2 - 1)

    x = np.linspace(0, length, num_pts)
    y = w0 * np.sqrt(1 + (x/zr)**2) / 2

    # Concatenate top and bottom vertices
    verts_top = np.vstack((x,y)).T
    verts_bot = np.vstack((x,-y)).T
    verts_bot = verts_bot[::-1]
    vertices = np.vstack((verts_top, verts_bot))
    
    return vertices

def euler_taper(w0:float, w1:float, theta_max:float, rad2dy:float, length=None, alpha=0.5, num_pts=100):
    """
    Generates vertices for Euler taper in clockwise order.

    Parameters:
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

    Returns:
        [N-by-2] ndarray : Shape vertices.

    Raises:
        ValueError: <w0> is less than <w1>.
        ValueError: <alpha> is less than 0 or greater than 1.
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

def _circular_curve(radius:float, angle_range:float, angle_start=0.0, direction='counterclockwise', num_pts=100):
    """
    Helper function to create a circular curve.
    """
    theta = np.linspace(angle_start, angle_start+angle_range, num_pts)
    theta = np.deg2rad(theta-90)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    x -= x[0]
    y -= y[0]

    theta += np.pi/2

    if direction == 'clockwise':
        reflect_angle = np.deg2rad(angle_start)
        x2 = x*np.cos(2*reflect_angle) + y*np.sin(2*reflect_angle)
        y2 = x*np.sin(2*reflect_angle) - y*np.cos(2*reflect_angle)
        x = x2
        y = y2
        theta = np.pi + 2*np.deg2rad(angle_start) - theta

    return x, y, np.rad2deg(theta)

def circular_arc(width:float, radius:float, angle_range:float, angle_start=0.0, direction='counterclockwise',  num_pts=100):
    """
    Generates a circular arc path.
    
    Parameters:
        width : float
            Path width.
        radius : float
            Arc center radius.
        angle_range : float
            Arc angular range in degrees.
        angle_start : float, optional
            Arc start angle in degrees. Zero degrees points to the +x direction.
        direction : str, optional
            Either "clockwise" or "counterclockwise".
        num_pts : int, optional
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.

    Raises:
        ValueError: <width> is larger than the diameter.
        ValueError: <angle_range> is zero, negative, or greater than 360.
        ValueError: <direction> is neither 'clockwise' or 'counterclockwise'.
    """
    # Check parameters
    if radius <= width/2:
        raise ValueError(
            'Input parameter <radius> must be greater than <width>/2.')
    if angle_range <= 0 or angle_range > 360:
        raise ValueError(
            'Input parameter <angle_range> must be between 0 and 360.')
    if angle_start >= 360:
        angle_start = angle_start % 360
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')
    if num_pts % 2 != 0:
        num_pts += 1

    xt, yt, theta = _circular_curve(
        radius=radius,
        angle_range=angle_range,
        angle_start=angle_start,
        direction=direction,
        num_pts=num_pts)
    
    # Add width to curve
    vertices = _thicken(x=xt, y=yt, theta=theta, width=width)

    return vertices

def circular_ring(width:float, radius:float, num_pts=100):
    """
    Generates a circular ring.
    
    Parameters:
        width : float
            Path width.
        radius : float
            Path center radius.
        num_pts : int, optional
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.
    """
    vertices = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=360, 
        num_pts=num_pts)
    return vertices

def circular_s_bend(width:float, radius:float, span:float, angle=None, reflect=False, num_pts=100):
    """
    Generates a circular S-bend.

    Parameters:
        width : float
            Path width.
        radius : float
            Path center radius of curvature.
        span : float
            Longitudinal distance between input and output.
        reflect : bool, optional
            Whether to reflect over longitudinal axis.
        num_pts : int, optional
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Rotated vertices.
    """
    if 2*radius < span:
        raise ValueError('Input parameter <radius> must be greater than <span>/2.')
    if num_pts % 2 != 0:
        num_pts += 1

    theta0 = np.rad2deg(np.arccos(np.sqrt((radius*span - span**2/4)/radius**2)))
    angle_range = 90 - theta0

    x1, y1, theta1 = _circular_curve(
        radius=radius,
        angle_range=angle_range,
        angle_start=0,
        direction='counterclockwise',
        num_pts=int(num_pts/2))
    x2, y2, _ = _circular_curve(
        radius=radius,
        angle_range=angle_range,
        angle_start=angle_range,
        direction='clockwise',
        num_pts=int(num_pts/2))
    theta2 = theta1[::-1]
    
    dx = max(x1) - min(x1)
    dy = max(y1) - min(y1)

    x2 += dx
    y2 += dy
    
    xt = np.hstack((x1, x2))
    yt = np.hstack((y1, y2))
    theta = np.hstack((theta1, theta2))

    # Add width to curve
    vertices = _thicken(x=xt, y=yt, theta=theta, width=width)

    # Flip upside down if true
    if reflect:
        vertices = _reflect(vertices, angle=0)

    return vertices

def circular_u_bend(width:float, span:float, direction='counterclockwise', num_pts=100):
    """
    Generates a circular 180 degree bend path.

    Parameters:
        width : float
            Path width.
        span : float
            Distance between input and output.
        direction : str
            Either "clockwise" or "counterclockwise".
        num_pts : int
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Rotated vertices.
    """
    vertices = circular_arc(
        width=width,
        radius=span/2,
        angle_range=180,
        angle_start=0,
        direction=direction,
        num_pts=num_pts)
    return vertices

def circular_l_bend(width:float, radius:float, direction='counterclockwise', num_pts=100):
    """
    Parameters:
        width : float
            Path width.
        span : float
            Distance between input and output.
        direction : str
            Either "clockwise" or "counterclockwise".
        num_pts : int
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Rotated vertices.
    """
    vertices = circular_arc(
        width=width,
        radius=radius,
        angle_range=90,
        angle_start=0,
        direction=direction,
        num_pts=num_pts)
    return vertices

def _clothoid_ode_rhs(state, t, kappa0, kappa1):
    """
    Helper function to set up Euler ODEs.
    """
    x, y, theta = state[0], state[1], state[2]
    return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1*t])

def _euler_curve(min_radius:float, angle_range:float, angle_start=0.0, num_pts=100):
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

    return x, y, np.rad2deg(theta)

def euler_arc(width:float, min_radius:float, angle_range:float, angle_start=0.0, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler (aka clothoidal) arc path.

    Parameters:
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

    Returns:
        [N-by-2] ndarray : Shape vertices.
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
    
    # Add width to curve
    vertices = _thicken(x=xt, y=yt, theta=theta, width=width)

    if direction == 'clockwise':
        vertices = _reflect(vertices=vertices, angle=angle_start)

    return vertices

def euler_s_bend(width:float, rad2dy:float, theta_max:float, span:float, length=None, reflect=False, num_pts=100):
    """
    Generates an Euler S-bend.

    Parameters:
        width : float
            Path width.
        rad2dy : float
            Ratio between minimum bend radius to vertical displacement.
        theta_max : float
            Maximum angle made by bend in degrees. Occurs at the turning point.
        span : float
            Transverse distance between input and output.
        length : int or float, optional
            Longitudinal distance between input and output. 
        reflect : bool, optional
            Whether to reflect over longitudinal axis.
        num_pts : int, optional
            Number of points on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.
    """
    if num_pts % 2 != 0:
        num_pts += 1

    min_radius = rad2dy * span/2

    # First part of curve before turning point
    x1, y1, theta1 = _euler_curve(
        min_radius=min_radius,
        angle_range=theta_max,
        num_pts=int(num_pts/2))

    # Last part of curve after turning point
    x2, y2, theta2 = -x1[::-1], -y1[::-1], theta1[::-1]

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

    # Add width to curve
    vertices = _thicken(x=xt, y=yt, theta=theta, width=width)

    # Flip upside down if true
    if reflect:
        vertices = _reflect(vertices=vertices, angle=0)

    return vertices

def euler_u_bend(width:float, span:float, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler 180 degree bend path.

    Parameters:
        width : float
            Path width.
        span : float
            Distance between input and output.
        direction : str
            Either "clockwise" or "counterclockwise".
        num_pts : float
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Shape vertices.
    """
    if num_pts % 2 != 0:
        num_pts += 1
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    # Simulated relationship between minimum bend radius and span.
    rad_to_span_ratio = 0.7263051699691657
    min_radius = rad_to_span_ratio * span/2

    half_num_pts = int(num_pts/2)

    verts1 = euler_arc(
        width=width, 
        min_radius=min_radius, 
        angle_range=90, 
        angle_start=0, 
        direction=direction, 
        num_pts=half_num_pts)
    
    verts2 = _reflect(vertices=verts1, angle=0)
    verts2 = _translate(vertices=verts2, dy=span)
    
    vertices = np.vstack((
        verts1[:half_num_pts-1], 
        verts2[half_num_pts-1::-1],
        verts2[-1:half_num_pts:-1],
        verts1[half_num_pts:]))

    return vertices

def euler_l_bend(width:float, min_radius:float, span=None, direction='counterclockwise', num_pts=100):
    """
    Generates an Euler 90 degree bend.

    Parameters:
        width : float
            Path width.
        min_radius : float
            Minimum radius of curvature.
        direction : str
            Either "clockwise" or "counterclockwise".
        num_pts : int
            Number of vertices on one side of the shape.

    Returns:
        [N-by-2] ndarray : Rotated vertices.
    """
    if num_pts % 2 != 0:
        num_pts += 1
    if direction != 'clockwise' and direction != 'counterclockwise':
        raise ValueError(
            'Input parameter <direction> must be either "clockwise" or "counterclockwise".')

    x1, y1, theta1 = _euler_curve(
        min_radius=min_radius, 
        angle_range=45, 
        num_pts=int(num_pts/2))
    
    dx = max(x1) - min(x1)
    dy = max(y1) - min(y1)

    # First half
    curve_pts_1 = np.vstack((x1, y1)).T

    # Second half
    curve_pts_2 = _reflect(
        vertices=curve_pts_1[::-1],
        angle=0)
    curve_pts_2 = _rotate(
        vertices=curve_pts_2,
        angle=-90)
    curve_pts_2 = _translate(
        vertices=curve_pts_2,
        dx=dx+dy,
        dy=dx+dy)
    
    # Combine both halves
    curve_pts = np.vstack((curve_pts_1, curve_pts_2))
    xt, yt = curve_pts[:,0], curve_pts[:,1]

    # Calculate angles
    theta2 = 90 - theta1[::-1]
    theta = np.hstack((theta1, theta2))
    
    # Resize if <span> is given
    if isinstance(span, (int, float)):
        x_error = span / xt[-1]
        y_error = span / yt[-1]
        xt *= x_error
        yt *= y_error

    # Add width to curve
    vertices = _thicken(x=xt, y=yt, theta=theta, width=width)
    
    return vertices