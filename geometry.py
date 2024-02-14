import numpy as np
import copy

def reflect_x(points):
    """
    Flips the vertices in the x-direction (reflection w/ respect to y-axis)

    Parameters:
        points
            type: [N-by-2] ndarray
            desc: vertices of the shape

    Returns:
        new_points
            type: [N-by-2] ndarray
            desc: flipped vertices
    """
    new_points = copy.copy(points)
    new_points[:,0] *= -1

    return new_points

def reflect_y(points):
    """
    Flips the vertices in the y-direction (reflection w/ respect to x-axis)

    Parameters:
        points
            type: [N-by-2] ndarray
            desc: vertices of the shape

    Returns:
        new_points
            type: [N-by-2] ndarray
            desc: flipped vertices
    """
    new_points = copy.copy(points)
    new_points[:,1] *= -1
    
    return new_points

def rotate(points, theta:float, origin=[0,0], unit='deg'):
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters:
        points
            type: [N-by-2] ndarray
            desc: vertices of the shape
        theta
            type: float
            desc: angle of rotation
        origin
            type: [1-by-2] array-like
            desc: point about which to rotate
        unit
            type: string
            desc: angle units ('deg' or 'rad')

    Returns:
        new_points
            type: [N-by-2] ndarray
            desc: rotated vertices
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
        w0
            type: float
            desc: width of one end of the taper
        w1
            type: float
            desc: width of the other end of the taper
        length
            type: float
            desc: distance between both ends of the taper

    Returns:
        points
            type: [N-by-2] ndarray
            desc: vertices of the linear taper in clockwise order
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
        w0
            type: float
            desc: width of one end of the taper
        w1
            type: float
            desc: width of the other end of the taper
        length
            type: float
            desc: distance between both ends of the taper
        n_points
            type: float
            desc: number of points with which to draw one side of the taper

    Returns:
        points
            type: [N-by-2] ndarray
            desc: vertices of the parabolic taper in clockwise order
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

def circular_arc(width:float, radius:float, theta:float, n_points=100):
    """
    Generates a circular path 
    """
    