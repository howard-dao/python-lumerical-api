import numpy as np
import copy

def reflect_x(points):
    """
    Flips the vertices in the x-direction (reflection w/ respect to y-axis)

    Parameters:
        points : [N-by-2] ndarray
            Vertices of the shape.

    Returns:
        new_points : [N-by-2] ndarray
            Flipped vertices.
    """
    new_points = copy.copy(points)
    new_points[:,0] *= -1

    return new_points

def reflect_y(points):
    """
    Flips the vertices in the y-direction (reflection w/ respect to x-axis)

    Parameters:
        points : [N-by-2] ndarray
            Vertices of the shape.

    Returns:
        new_points : [N-by-2] ndarray
            Flipped vertices.
    """
    new_points = copy.copy(points)
    new_points[:,1] *= -1
    
    return new_points

def rotate(points, theta:float, origin=[0,0], unit='deg'):
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters:
        points : [N-by-2] ndarray
            Vertices of the shape.
        theta : float
            Angle of rotation.
        origin : [1-by-2] array-like
            Point about which to rotate.
        unit : str
            Angle units ('deg' or 'rad').

    Returns:
        new_points : [N-by-2] ndarray
            Rotated vertices.
    """

    ox,oy = origin

    px,py = zip(*points)

    if unit=='deg':
        theta *= np.pi/180

    qx = ox + np.cos(theta) * np.subtract(px,ox) - np.sin(theta) * np.subtract(py,oy)
    qy = oy + np.sin(theta) * np.subtract(px,ox) + np.cos(theta) * np.subtract(py,oy)

    new_points = [(qx,qy) for qx, qy in zip(qx, qy)]
    new_points = np.array(new_points)

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
            Vertices of the linear taper in clockwise order,
            starting from the top left vertex.
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
            Vertices of the parabolic taper in clockwise order,
            starting from the top left vertex.
    """
    a = 4 * length / (w1**2 - w0**2)
    c = a * w0**2 / 4

    x = np.linspace(0, length, n_points)
    y = ((x + c) / a) ** 0.5

    points_top = [(xp, yp) for xp, yp in zip(x, y)]
    points_bot = [(xp, -yp) for xp, yp in zip(x, y)]
    points_bot = list(reversed(points_bot))
    
    points = np.array(points_top + points_bot)
    
    return points

def circular_arc(width:float, radius:float, angle_range:float, angle_start=0, n_points=100):
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
            Arc start angle in degrees.
        n_points : int, optional
            Number of points.

    Returns:
        points : [N-by-2] ndarray

    Raises:
        ValueError: Width is larger than the diameter.
        ValueError: Angular range is zero, negative, or greater than 360.
    """
    inner_radius = radius - width/2
    outer_radius = radius + width/2

    if inner_radius <= 0:
        raise ValueError(
            'Input parameter <width> must be less than 2*<radius>.')
    if angle_range <= 0 or angle_range > 360:
        raise ValueError(
            'Input parameter <angle_range> must be between 0 and 360.')
    if angle_start >= 360:
        angle_start = angle_start % 360

    theta = np.linspace(angle_start, angle_start+angle_range, round(n_points/2))
    theta = theta * np.pi / 180
    x_inner = inner_radius * np.cos(theta)
    y_inner = inner_radius * np.sin(theta)

    x_outer = outer_radius * np.cos(theta)
    y_outer = outer_radius * np.sin(theta)

    points_inner = [(xp, yp) for xp, yp in zip(x_inner, y_inner)]
    points_outer = [(xp, yp) for xp, yp in zip(x_outer, y_outer)]
    points_outer = list(reversed(points_outer))

    points = np.array(points_inner + points_outer)

    return points