"""
Python module containing basic functions for shape generation.
Author(s): Howard Dao
"""

import numpy as np
from scipy.integrate import odeint
import copy

def _translate(vertices:np.ndarray, dx:float=0.0, dy:float=0.0) -> np.ndarray[float]:
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

def _reflect(vertices:np.ndarray, angle:float) -> np.ndarray[float]:
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

def _rotate(vertices:np.ndarray, angle:float, origin:tuple[float,float]=(0.0,0.0)) -> np.ndarray[float]:
    """
    Rotates a shape counterclockwise about an origin point.

    Parameters
    ----------
    vertices : ndarray
        Shape vertices.
    angle : float
        Angle of rotation in degrees.
    origin : [1-by-2] tuple
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

def _mirror(x:np.ndarray, y:np.ndarray, axis:str) -> np.ndarray[float]:
    """
    Duplicates a set of vertices, reflects it in a given direction, then concatenates it with the first set.

    Parameters
    ----------
    x : ndarray
        x data.
    y : ndarray
        y data.
    axis : str
        Direction in which to mirror vertices. Must be either "x" or "y".

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    
    Raises
    ----------
    ValueError
        `axis` is neither "x" nor "y".
    """
    v1 = np.vstack((x,y)).T

    if axis == 'x':
        v2 = np.vstack((-x,y)).T
        v2 = v2[::-1]
    elif axis == 'y':
        v2 = np.vstack((x,-y)).T
        v2 = v2[::-1]
    else:
        raise ValueError(f'Input parameter <axis> is neither "x" or "y". It was given "{axis}".')
    vertices = np.vstack((v1,v2))

    return vertices

def _thicken(curve:np.ndarray, width:float, angle_start:float, angle_final:float):
    """
    Adds width to a curve.

    Parameters
    ----------
    curve : ndarray
        Curve data.
    width : float
        Path width.
    angle_start : float
        Start angle.
    angle_final : float
        Final angle.

    Returns
    ----------
    vertices : ndarray
        Shape vertices.
    """
    if width <= 0:
        raise ValueError('Input parameter <width> must be greater than zero.')
    
    x0 = curve[:,0]
    y0 = curve[:,1]

    theta = np.zeros(x0.shape)
    for idx in range(len(theta)-2):
        # Backward vector
        dx_backward = x0[idx] - x0[idx+1]
        dy_backward = y0[idx] - y0[idx+1]
        v_backward = np.array([dx_backward, dy_backward])
        magnitude_backward = np.linalg.norm(v_backward)
        phi_backward = np.arctan2(dy_backward, dx_backward)
        if phi_backward < 0:
            phi_backward += 2*np.pi

        # Forward vector
        dx_forward = x0[idx+2] - x0[idx+1]
        dy_forward = y0[idx+2] - y0[idx+1]
        v_forward = np.array([dx_forward, dy_forward])
        magnitude_forward = np.linalg.norm(v_forward)
        phi_forward = np.arctan2(dy_forward, dx_forward)
        if phi_forward < 0:
            phi_forward += 2*np.pi

        dot_product = np.dot(v_backward, v_forward)

        # Difference between forward and backward vectors (in radians)
        a = dot_product / (magnitude_backward*magnitude_forward)
        a = np.max([-1,a])
        a = np.min([1,a])
        angular_difference = np.arccos(a)

        theta[idx+1] = phi_forward + angular_difference/2 - np.pi/2

    theta[0] = np.deg2rad(angle_start)
    theta[-1] = np.deg2rad(angle_final)

    x1 = x0 + (width/2)*np.cos(theta + np.pi/2)
    y1 = y0 + (width/2)*np.sin(theta + np.pi/2)

    x2 = x0 + (width/2)*np.cos(theta - np.pi/2)
    y2 = y0 + (width/2)*np.sin(theta - np.pi/2)

    x = np.hstack((x1, x2[::-1]))
    y = np.hstack((y1, y2[::-1]))
    vertices = np.vstack((x,y)).T

    return vertices

def _stitch(*args, fraction:float=0.5) -> np.ndarray[float]:
    """
    Stitches two or more shapes together.

    Parameters
    ----------
    args
        Vertices for all shapes to be stitched together.
    fraction : float, optional
        Fraction of shape to be stitched.
    Returns
    ----------
    vertices : ndarray
        Shape vertices.

    Raises
    ----------
    TypeError
        At least one of the arguments is not an ndarray.
    ValueError
        At least one of the arguments is not an N-by-2 ndarray.
    ValueError
        Only one shape was given.
    ValueError
        `fraction` is outside range (0,1).
    """
    if any(not isinstance(arg, np.ndarray) for arg in args):
        raise TypeError('At least 1 of the arguments is not an array of points.')
    if any(arg.shape[1] != 2 for arg in args):
        raise ValueError('At least 1 of the arguments is not a set of (x,y) data.')
    if len(args) == 1:
        raise ValueError('Need at least 2 sets of vertices. Given only 1.')
    if fraction >= 1 or fraction <= 0:
        raise ValueError(f'Input parameter <fraction> must be between 0 and 1. It was given {fraction}.')

    x = np.array([])
    y = np.array([])

    for idx, arg in enumerate(args):
        num_pts = len(arg[:,0])
        frac_num_pts = int(num_pts * fraction)

        x = np.hstack((x, arg[:frac_num_pts,0]))
        y = np.hstack((y, arg[:frac_num_pts,1]))

    for idx, arg in enumerate(args[::-1]):
        num_pts = len(arg[:,0])
        frac_num_pts = int(num_pts * fraction)

        x = np.hstack((x, arg[frac_num_pts:,0]))
        y = np.hstack((y, arg[frac_num_pts:,1]))

    vertices = np.vstack((x,y)).T

    return vertices

def _circular_curve(radius:float, angle_range:float, angle_start:float=0.0, direction:str='counterclockwise', num_pts:int=100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    vertices = np.vstack((x,y)).T

    return vertices, np.rad2deg(theta)

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

def _euler_curve(min_radius:float, angle_range:float, angle_start:float=0.0, num_pts:int=100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    ----------
    x : ndarray
        x data.
    y : ndarray
        y data.
    theta : ndarray
        Angle of the line tangent to the curve, in degrees.
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

    curve = np.vstack((x,y)).T

    return curve, np.rad2deg(theta)