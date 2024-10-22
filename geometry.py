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

    Parameters
    ----------
    vertices : ndarray
        Shape vertices.
    dx : float, optional
        Distance along x.
    dy : float, optional
        Distance along y.

    Returns
    ----------
    new_vertices : ndarray
        Shape vertices.
    """
    new_vertices = copy.copy(vertices)
    new_vertices[:,0] += dx
    new_vertices[:,1] += dy

    return new_vertices

def _reflect(vertices:np.ndarray, angle:float):
    """
    Reflects vertices with respect to a given angle.

    Parameters
    ----------
    vertices : ndarray
        Shape vertices.
    angle : float
        Angle over which to reflect vertices in degrees.

    Returns
    ----------
    new_vertices : ndarray
        Shape vertices.
    """
    angle = np.deg2rad(angle)
    matrix = np.array([[np.cos(2*angle), np.sin(2*angle)],
                       [np.sin(2*angle), -np.cos(2*angle)]])
    new_vertices = np.transpose(np.matmul(matrix, vertices.T))

    return new_vertices

def _rotate(vertices:np.ndarray, angle:float, origin=[0,0]):
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters
    ----------
    vertices : ndarray
        Shape vertices.
    angle : float
        Angle of rotation in degrees.
    origin : [1-by-2] array-like
        Point about which to rotate.

    Returns
    ----------
    new_vertices : ndarray
        Shape vertices.
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

    Parameters
    ----------
    x : ndarray
        x data.
    y : ndarray
        y data.
    theta : ndarray
        Angle of the line tangent to the curve, in degrees.
    width : float
        Path width.

    Returns
    ----------
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

def _stitch(*args):
    """
    Stitches two or more shapes together.

    Parameters
    ----------
    args
        Vertices for all shapes to be stitched together.
    """
    if len(args) == 1:
        raise ValueError('Need at least 2 sets of vertices. Given only 1.')
    if any(not isinstance(arg, np.ndarray) for arg in args):
        raise TypeError('At least 1 of the parameters is not an array of points.')

    x = np.array([])
    y = np.array([])

    for idx, arg in enumerate(args):
        num_pts = len(arg[:,0])
        half_num_pts = int(num_pts/2)

        x = np.hstack((x, arg[:half_num_pts,0]))
        y = np.hstack((y, arg[:half_num_pts,1]))

    for idx, arg in enumerate(args[::-1]):
        num_pts = len(arg[:,0])
        half_num_pts = int(num_pts/2)

        x = np.hstack((x, arg[half_num_pts:,0]))
        y = np.hstack((y, arg[half_num_pts:,1]))

    vertices = np.vstack((x,y)).T

    return vertices

def _circular_curve(radius:float, angle_range:float, angle_start=0.0, direction='counterclockwise', num_pts=100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to create a circular curve.

    Parameters
    ----------
    radius : float
        Radius of curvature.
    angle_range : float
        Angular range in degrees.
    angle_start : float, optional
        Initial angle in degrees. Zero degrees points to the +x direction.
    direction : str, optional
        Either "clockwise" or "counterclockwise".
    num_pts : int, optional
        Number of vertices.
    
    Returns
    ----------
    x : ndarray
        x data.
    y : ndarray
        y data.
    theta : ndarray
        Angle of the line tangent to the curve.
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

def _clothoid_ode_rhs(state:np.ndarray, t:np.ndarray, kappa0:float, kappa1:float):
    """
    Helper function to set up Euler ODEs.

    Parameters
    ----------
    state : ndarray
        Array of initial conditions.
    t : ndarray
        Parametric variable.
    kappa0 : float
        Curvature.
    kappa1 : float
        Rate of change of curvature. 
    """
    x, y, theta = state[0], state[1], state[2]
    return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1*t])

def _euler_curve(min_radius:float, angle_range:float, angle_start=0.0, num_pts=100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to create an Euler curve.

    Parameters
    ----------
    min_radius : float
        Minimum radius of curvature.
    angle_range : float
        Angular range in degrees.
    angle_start : float, optional
        Initial angle in degrees. Zero degrees points to the +x direction.
    num_pts : int, optional
        Number of vertices.
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