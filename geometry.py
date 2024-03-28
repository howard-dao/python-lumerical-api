import numpy as np
import copy

def _translate(points, dx:float, dy:float):
    """
    Translates vertices along either x and y directions.

    Parameters:
        points : [N-by-2] ndarray
            Vertices of the shape.
        dx : float
            Distance along x.
        dy : float
            Distance along y.

    Returns:
        [N-by-2] array : Translated vertices.
    """
    new_points = copy.copy(points)
    new_points[:,0] += dx
    new_points[:,1] += dy

    return new_points

def reflect(points, angle:float):
    """
    Reflects vertices with respect to a given angle.

    Parameters:
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

def rotate(points, angle:float, origin=[0,0], unit='deg'):
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters:
        points : [N-by-2] ndarray
            Vertices of the shape.
        angle : float
            Angle of rotation.
        origin : [1-by-2] array-like
            Point about which to rotate.
        unit : str
            Angle units ('deg' or 'rad').

    Returns:
        [N-by-2] ndarray : Rotated vertices.
    """
    if unit == 'deg':
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

    points = [(x,y) for x, y in zip(x, y)]
    points = np.array(points)

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
    y = ((x + c) / a) ** 0.5

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
        points = reflect(points=points, angle=angle_start)

    return points

def circular_s_bend(width:float, radius:float, span:float, n_points=100):
    """
    Generates a circular S-bend.

    Parameters:
        width : float
            Arc width.
        radius : float
            Arc center radius of curvature.
        span : float

        n_points : int, optional
    """
    theta1 = np.rad2deg(np.arccos(np.sqrt((radius*span - span**2/4)/radius**2)))
    angle_range = 90 - theta1

    points1 = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=angle_range, 
        angle_start=0, 
        direction='clockwise', 
        n_points=n_points/2)
    points2 = circular_arc(
        width=width, 
        radius=radius, 
        angle_range=angle_range, 
        angle_start=-angle_range, 
        direction='counterclockwise', 
        n_points=n_points/2)
    
    # Align the second arc with the first arc.
    theta2 = np.deg2rad(90-angle_range)
    length = 2 * radius * np.sqrt(1 - np.sin(theta2)**2)
    points2 = _translate(points2, dx=length/2, dy=-span/2)

    points = np.vstack((points1, points2))

    return points