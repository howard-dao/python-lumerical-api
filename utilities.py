"""
Python wrapper for Lumerical FDTD and MODE
Author(s): Howard Dao
"""

import numpy as np
from scipy.constants import electron_volt

class LumericalBase():
    def __init__(self, lum) -> None:
        self.lum = lum
    
    def _match_material(self, material):
        """
        Matches the given material to Lumerical's database and assigns it to the object.
        If a number, set the material's refractive index instead.

        Parameters
        ----------
        material : float or str
            Name or refractive index of the material.
        """
        match material:
            case str():
                if self.lum.materialexists(material):
                    self.lum.set('material', material)
                else:
                    raise ValueError('Input parameter <material> does not match any material in database.')
            case int() | float():
                self.lum.set('index', material)
            case _:
                raise TypeError('Input parameter <material> must be either a string, integer, or float.')
            
    def _set_x_coords(self, x=None, x_span=None, x_min=None, x_max=None):
        """
        Helper function for setting either the (x, x span) or (x min, x max).
        """
        if all(isinstance(arg, (int,float)) for arg in (x, x_span)):
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
        elif all(isinstance(arg, (int,float)) for arg in (x_min, x_max)):
            self.lum.set('x min', x_min)
            self.lum.set('x max', x_max)
        else:
            raise ValueError('Input parameters <x>, <x_span>, <x_min>, <x_max> are not given.')
        
    def _set_y_coords(self, y=None, y_span=None, y_min=None, y_max=None):
        """
        Helper function for setting either the (y, y span) or (y min, y max).
        """
        if all(isinstance(arg, (int,float)) for arg in (y, y_span)):
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
        elif all(isinstance(arg, (int,float)) for arg in (y_min, y_max)):
            self.lum.set('y min', y_min)
            self.lum.set('y max', y_max)
        else:
            raise ValueError('Input parameters <y>, <y_span>, <y_min>, <y_max> are not given.')
    
    def _set_z_coords(self, z=None, z_span=None, z_min=None, z_max=None):
        """
        Helper function for setting either the (z, z span) or (z min, z max).
        """
        if all(isinstance(arg, (int,float)) for arg in (z, z_span)):
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif all(isinstance(arg, (int,float)) for arg in (z_min, z_max)):
            self.lum.set('z min', z_min)
            self.lum.set('z max', z_max)
        else:
            raise ValueError('Input parameters <z>, <z_span>, <z_min>, <z_max> are not given.')
            
    def _draw_3D_box(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None):
        """
        Helper function for drawing a 3D region.
        """
        self._set_x_coords(x=x, x_span=x_span, x_min=x_min, x_max=x_max)
        self._set_y_coords(y=y, y_span=y_span, y_min=y_min, y_max=y_max)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        
    def _draw_2D_box_x(self, x:float, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None):
        """
        Helper function for drawing a 2D box normal to the x-axis.
        """
        self.lum.set('x', x)      
        self._set_y_coords(y=y, y_span=y_span, y_min=y_min, y_max=y_max)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        
    def _draw_2D_box_y(self, y:float, x=None, x_span=None, x_min=None, x_max=None, z=None, z_span=None, z_min=None, z_max=None):
        """
        Helper function for drawing a 2D box normal to the y-axis.
        """ 
        self._set_x_coords(x=x, x_span=x_span, x_min=x_min, x_max=x_max)
        self.lum.set('y', y)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        
    def _draw_2D_box_z(self, z:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None):
        """
        Helper function for drawing a 2D box normal to the z-axis.
        """     
        self._set_x_coords(x=x, x_span=x_span, x_min=x_min, x_max=x_max)
        self._set_y_coords(y=y, y_span=y_span, y_min=y_min, y_max=y_max)
        self.lum.set('z', z)

    def add_rect(self, material, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, mesh_order=2, name='Rectangle', alpha=0.5):
        """
        Adds a rectangle object in the simulation.

        Parameters
        ----------
        material : float or str
            If float, sets material index; if string, sets material to library material.
        mesh_order : int, optional
            Priority number with which to draw the object. Lower means higher priority.
        name : str, optional
            Object name.
        alpha : float, optional
            Object render opacity.
        """
        self.lum.addrect()
        self.lum.set('name', name)

        # Geometry
        self._draw_3D_box(
            x=x, x_span=x_span, x_min=x_min, x_max=x_max, 
            y=y, y_span=y_span, y_min=y_min, y_max=y_max, 
            z=z, z_span=z_span, z_min=z_min, z_max=z_max)

        # Material
        self._match_material(material=material)
        self.lum.set('alpha', alpha)
        self.lum.set('override mesh order from material database', True)
        self.lum.set('mesh order', mesh_order)

    def add_circle(self, x:float, y:float, radius:float, axis:str, theta:float, material, z=None, z_span=None, z_min=None, z_max=None, mesh_order=2, name='Circle', alpha=0.5):
        """
        Adds a cylinder object in the simulation.

        Parameters
        ----------
        radius : float
            Radius in meters.
        axis : str
            Axis of rotation ("x", "y", or "z").
        theta : float
            Rotation angle in degrees.
        material : float or str
            If float, sets material index; if string, sets material to library material.
        mesh_order : int, optional
            Priority number with which to draw the object. Lower means higher priority.
        name : str, optional
            Object name.
        alpha : float, optional
            Object render opacity.
        """
        self.lum.addcircle()
        self.lum.set('name', name)

        # Geometry
        self.lum.set('x', x)
        self.lum.set('y', y)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        self.lum.set('radius', radius)
        self.lum.set('first axis', axis)
        self.lum.set('rotation 1', theta)

        # Material
        self._match_material(material=material)
        self.lum.set('alpha', alpha)
        self.lum.set('override mesh order from material database', True)
        self.lum.set('mesh order', mesh_order)

    def add_ring(self, x:float, y:float, r_out:float, r_in:float, material, z=None, z_span=None, z_min=None, z_max=None, mesh_order=2, name='Ring', alpha=0.5):
        """
        Adds a ring object in the simulation.

        Parameters
        ----------
        r_out : float
            Outer radius in meters.
        r_in : float
            Inner radius in meters.
        material : float or str
            If float, sets material index; if string, sets material to library material.
        mesh_order : int, optional
            Priority number with which to draw the object. Lower means higher priority.
        name : str, optional
            Object name.
        alpha : float, optional
            Object render opacity.
        """
        self.lum.addring()
        self.lum.set('name', name)

        # Geometry
        self.lum.set('x', x)
        self.lum.set('y', y)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)

        if r_out <= r_in:
            raise ValueError('Input argument <r_out> must be greater than <r_in>.')
        self.lum.set('outer radius', r_out)
        self.lum.set('inner radius', r_in)

        # Material
        self._match_material(material=material)
        self.lum.set('alpha', alpha)
        self.lum.set('override mesh order from material database', True)
        self.lum.set('mesh order', mesh_order)

    def add_poly(self, x:float, y:float, vertices, material, z=None, z_span=None, z_min=None, z_max=None, mesh_order=2, name='Polygon', alpha=0.5):
        """
        Adds a polygon object in the simulation.

        Parameters
        ----------
        vertices : [N-by-2] array
            Vertices defining polygon shape.
        material : float or str
            If float, sets material index; if string, sets material to library material.
        mesh_order : int, optional
            Priority number with which to draw the object. Lower means higher priority.
        name : str, optional
            Object name.
        alpha : float, optional
            Object render opacity.
        """
        self.lum.addpoly()
        self.lum.set('name', name)

        # Geometry
        self.lum.set('vertices', vertices)
        self.lum.set('x', x)
        self.lum.set('y', y)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)

        # Material
        self._match_material(material=material)
        self.lum.set('alpha', alpha)
        self.lum.set('override mesh order from material database', True)
        self.lum.set('mesh order', mesh_order)

    def add_mesh(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, dx=None, dy=None, dz=None, structure=None, name='Mesh'):
        """
        Adds a mesh override to a specific area or structure in the simulation.

        Parameters
        ----------
        dx : float, optional
            x discretization in meters.
        dy : float, optional
            y discretization in meters.
        dz : float, optional
            z discretization in meters.
        structure : str, optional
            Simulation object on which to apply mesh override.
        name : str, optional
            Mesh override name.
        """
        self.lum.addmesh()
        self.lum.set('name', name)

        # Geometry
        dimensions = (x, x_span, x_min, x_max, y, y_span, y_min, y_max, z, z_span, z_min, z_max)
        if not all(arg is None for arg in dimensions):
            self._draw_3D_box(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max, 
                y=y, y_span=y_span, y_min=y_min, y_max=y_max, 
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)

        # If <structure> is specified as a string, override the mesh of the structure.
        # This only takes effect if <structure> is exactly equal to the name of the object
        # that's being overridden, not the name of the mesh itself.
        if isinstance(structure, str):
            self.lum.set('based on a structure', True)
            self.lum.set('structure', structure)

        # Sets the mesh overrides if they are specified
        if isinstance(dx, float):
            self.lum.set('override x mesh', True)
            self.lum.set('dx', dx)
        if isinstance(dy, float):
            self.lum.set('override y mesh', True)
            self.lum.set('dy', dy)
        if isinstance(dz, float):
            self.lum.set('override z mesh', True)
            self.lum.set('dz', dz)

    def add_index_monitor(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D X-normal', name='Index Monitor'):
        """
        Adds an index monitor.

        Parameters
        ----------
        monitor_type : str, optional
            Monitor orientation.
        name : str, optional
            Monitor name.
        """
        self.lum.addindex()

        # General
        self.lum.set('name', name)

        # Geometry
        if monitor_type == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{monitor_type}".')

class LumericalFDTD(LumericalBase):
    """
    Class for FDTD simulation.
    """
    def __init__(self, lum) -> None:
        super().__init__(lum)

    def _mode_select(self, mode_selection='fundamental mode', modes=None):
        """
        Helper function for selecting modes.
        """
        settings = ('fundamental mode', 'fundamental TE mode', 'fundamental TM mode', 'user select')

        if mode_selection not in settings:
            raise ValueError(f'Input parameter <mode_selection> must be one of the following strings: "fundamental mode", "fundamental TE mode", "fundamental TM mode", or "user select". It was given "{mode_selection}".')
        self.lum.set('mode selection', mode_selection)
        if mode_selection == 'user select' and isinstance(modes, np.ndarray):
            if any(not isinstance(mode_item, np.int32) for mode_item in modes):
                raise ValueError('Input parameter <modes> must be a numpy ndarray containing only integers (numpy int32). At least one value is a non-integer.')
            self.lum.set('selected mode number', modes)

    def add_fdtd_3D(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, background_material=1.0, x_min_bc='PML', x_max_bc='PML', y_min_bc='PML', y_max_bc='PML', z_min_bc='PML', z_max_bc='PML', mesh_accuracy=2, simulation_time=1000e-15):
        """
        Adds a 3D FDTD simulation region.

        Parameters
        ----------
        background_material : str or float
            If float, sets material index; if string, sets material to library material.
        mesh_accuracy : int, optional
            Mesh resolution.
        simulation_time : float, optional
            Maximum duration of a simulation in seconds.
        """
        self.lum.addfdtd()

        # General
        self.lum.set('dimension', '3D')
        self.lum.set('simulation time', simulation_time)
        match background_material:
            case str():
                if self.lum.materialexists(background_material):
                    self.lum.set('background material', background_material)
                else:
                    raise ValueError('Input parameter <background_material> does not match any material in database.')
            case int() | float():
                self.lum.set('index', background_material)
            case _:
                raise TypeError('Input parameter <background_material> must be either a string, integer, or float.')

        # Geometry
        self._draw_3D_box(
            x=x, x_span=x_span, x_min=x_min, x_max=x_max, 
            y=y, y_span=y_span, y_min=y_min, y_max=y_max, 
            z=z, z_span=z_span, z_min=z_min, z_max=z_max)

        # Boundary Conditions
        self.lum.set('x min bc', x_min_bc)
        self.lum.set('x max bc', x_max_bc)
        self.lum.set('y min bc', y_min_bc)
        self.lum.set('y max bc', y_max_bc)
        self.lum.set('z min bc', z_min_bc)
        self.lum.set('z max bc', z_max_bc)

        # Mesh Settings
        self.lum.set('mesh accuracy', mesh_accuracy)

    def add_fdtd_2D(self, z:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, background_material=1.0, x_min_bc='PML', x_max_bc='PML', y_min_bc='PML', y_max_bc='PML', mesh_accuracy=2, simulation_time=1000e-15):
        """
        Adds a 2D FDTD simulation area.

        Parameters
        ----------
        mesh_accuracy : int, optional
            Mesh resolution.
        simulation_time : float, optional
            Maximum duration of a simulation in seconds.
        """
        self.lum.addfdtd()

        # General
        self.lum.set('dimension', '2D')
        self.lum.set('simulation time', simulation_time)
        match background_material:
            case str():
                if self.lum.materialexists(background_material):
                    self.lum.set('background material', background_material)
                else:
                    raise ValueError('Input parameter <background_material> does not match any material in database.')
            case int() | float():
                self.lum.set('index', background_material)
            case _:
                raise TypeError('Input parameter <background_material> must be either a string, integer, or float.')

        # Geometry
        self._draw_2D_box_z(
            x=x, x_span=x_span, x_min=x_min, x_max=x_max,
            y=y, y_span=y_span, y_min=y_min, y_max=y_max,
            z=z)

        # Boundary Conditions
        self.lum.set('x min bc', x_min_bc)
        self.lum.set('x max bc', x_max_bc)
        self.lum.set('y min bc', y_min_bc)
        self.lum.set('y max bc', y_max_bc)

        # Mesh Settings
        self.lum.set('mesh accuracy', mesh_accuracy)

    def set_global_monitors(self, center_wl:float, wl_span:float, use_wl_spacing=True, num_pts=21):
        """
        Set global monitor settings.

        Parameters
        ----------
        center_wl : float
            Center wavelength in meters.
        wl_span : float
            Wavelength span in meters.
        use_wl_spacing : bool, optional
            True for equal wavelength intervals. False for equal frequency intervals.
        num_pts : int, optional
            Number of wavelength points.
        """
        if use_wl_spacing:
            self.lum.setglobalmonitor('use wavelength spacing', True)
        else:
            self.lum.setglobalmonitor('use wavelength spacing', False)
            
        self.lum.setglobalmonitor('wavelength center', center_wl)
        self.lum.setglobalmonitor('wavelength span', wl_span)
        self.lum.setglobalmonitor('frequency points', num_pts)

    def add_power_monitor(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D X-normal', name='Power Monitor'):
        """
        Adds a field and power monitor in the simulation.

        Parameters
        ----------
        monitor_type : str, optional
            Monitor orientation.
        name : str, optional
            Monitor name.
        """
        self.lum.addpower()

        # General
        self.lum.set('name', name)

        # Geometry
        self.lum.set('monitor type', monitor_type)

        if monitor_type == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{monitor_type}".')

    def add_expansion_monitor(self, center_wl:float, wl_span:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D X-normal', mode_selection='fundamental mode', modes=np.array([1]), num_pts=21, name='Expansion Monitor'):
        """
        Adds a mode expansion monitor in the simulation.

        Parameters
        ----------
        monitor_type : str, optional
            Monitor orientation.
        mode_selection : str, optional
            Either "fundamental mode" or "user select".
        modes : ndarray, optional
            Numpy array of mode numbers, all of which must be integers.
        num_pts : int, optional
            Number of wavelength points.
        name : str, optional
            Monitor name.
        """
        self.lum.addmodeexpansion()

        # General
        self.lum.set('name', name)

        # Geometry
        self.lum.set('monitor type', monitor_type)

        if monitor_type == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{monitor_type}".')

        # Mode expansion
        self._mode_select(mode_selection=mode_selection, modes=modes)
        
        self.lum.set('override global monitor settings', True)
        self.lum.set('use wavelength spacing', True)

        using_source = self.lum.get('use source limits')
        if not using_source:
            self.lum.set('center wavelength', center_wl)
            self.lum.set('wavelength span', wl_span)

        self.lum.set('frequency points', num_pts)

    def set_expansion(self, expansion_monitor:str, power_monitor:str, port='port'):
        """
        Associates a power monitor with a mode expansion monitor.

        Parameters
        ----------
        expansion_monitor : str
            Name of the expansion monitor, must exist in the simulation.
        power_monitor : str
            Name of the DFT monitor, must exist in the simulation.
        port : str, optional
            Name of expansion port from which data is collected.
        """
        self.lum.select(expansion_monitor)
        self.lum.setexpansion(port, power_monitor)

    def add_movie_monitor(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D X-normal', name='Movie Monitor'):
        """
        Adds a movie monitor.

        Parameters
        ----------
        monitor_type : str, optional
            Monitor orientation.
        name : str, optional
            Monitor name.
        """
        self.lum.addmovie()

        # General
        self.lum.set('name', name)

        # Geometry
        self.lum.set('monitor type', monitor_type)

        if monitor_type == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{monitor_type}".')

    def add_mode(self, center_wl:float, wl_span:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, axis='x-axis', direction='forward', mode_selection='fundamental mode', modes=np.array([1])):
        """
        Adds a mode source in the simulation. Depending on the injection axis, 
        the span along that axis will be disabled.

        Parameters
        ----------
        center_wl : float
            Center wavelength in meters.
        wl_span : float
            Wavelength span in meters.
        axis : str
            Propagation axis. May be either "x-axis", "y-axis", or "z_axis".
        direction : str, optional
            Either "forward" or "backward".
        mode_selection : str, optional
            Either "fundamental mode" or "user select".
        modes : ndarray, optional
            Numpy array of mode numbers, all of which must be integers.
        """
        self.lum.addmode()

        # General
        self.lum.set('injection axis', axis)
        self.lum.set('direction', direction)
        self._mode_select(mode_selection=mode_selection, modes=modes)

        # Geometry
        if axis == 'x-axis':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif axis == 'y-axis':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif axis == 'z-axis':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <axis> must be either "x-axis", "y-axis", or "z-axis". It was given "{axis}".')

        # Frequency/Wavelength
        self.lum.set('center wavelength', center_wl)
        self.lum.set('wavelength span', wl_span)

    def add_gaussian(self, center_wl:float, wl_span:float, radius:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, angle=0.0, axis='x', direction='forward', polarization_angle=0.0, distance=0.0):
        """
        Adds a Gaussian source. Uses waist radius.

        Parameters
        ----------
        center_wl : float
            Center wavelength in meters.
        wl_span : float
            Wavelength span in meters.
        radius : float
            Waist radius in meters.
        angle : float
            Angular direction in degrees.
        axis : str, optional
            Propagation axis ("x", "y", or "z").
        direction : str, optional
            Either "forward" or "backward".
        distance : float, optional
            Distance from waist in meters. Positive distance corresponds to a diverging beam, and negative distance corresponds to a converging beam.
        """
        self.lum.addgaussian()

        # General
        self.lum.set('injection axis', axis)
        self.lum.set('direction', direction)
        self.lum.set('angle theta', angle)
        self.lum.set('polarization angle', polarization_angle)

        # Geometry
        if axis == 'x':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif axis == 'y':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif axis == 'z':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <axis> must be either "x", "y", or "z". It was given "{axis}".')
        
        # Frequency/Wavelength
        self.lum.set('center wavelength', center_wl)
        self.lum.set('wavelength span', wl_span)

        # Beam options
        self.lum.set('use scalar approximation', True)
        self.lum.set('waist radius w0', radius)
        self.lum.set('distance from waist', distance)
    
class LumericalMODE(LumericalBase):
    """
    Class for FDE, EME, and varFDTD simulations.
    """
    def __init__(self, lum) -> None:
        super().__init__(lum)

    def add_fde(self, wl:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, x_min_bc='Metal', x_max_bc='Metal', y_min_bc='Metal', y_max_bc='Metal', z_min_bc='Metal', z_max_bc='Metal', solver_type='2D X normal', num_modes=4):
        """
        Adds a Finite Difference Eigenmode (FDE) solver region in the simulation.
        """
        self.lum.addfde()

        # Geometry
        self.lum.set('solver type', solver_type)
        if solver_type == '2D X normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
            self.lum.set('y min bc', y_min_bc)
            self.lum.set('y max bc', y_max_bc)
            self.lum.set('z min bc', z_min_bc)
            self.lum.set('z max bc', z_max_bc)
        elif solver_type == '2D Y normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
            self.lum.set('x min bc', x_min_bc)
            self.lum.set('x max bc', x_max_bc)
            self.lum.set('z min bc', z_min_bc)
            self.lum.set('z max bc', z_max_bc)
        elif solver_type == '2D Z normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
            self.lum.set('x min bc', x_min_bc)
            self.lum.set('x max bc', x_max_bc)
            self.lum.set('y min bc', y_min_bc)
            self.lum.set('y max bc', y_max_bc)
        else:
            raise ValueError(f'Input parameter <solver_type> must be either "2D X normal", "2D Y normal", or "2D Z normal". It was given "{solver_type}".')
        
        self.lum.setanalysis('wavelength', wl)
        self.lum.setanalysis('number of trial modes', num_modes)

    def add_varfdtd(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, simulation_time=1000e-15, x0=0.0, y0=0.0, index_method='variational', polarization='E mode (TE)', mesh_accuracy=2, x_min_bc='PML', x_max_bc='PML', y_min_bc='PML', y_max_bc='PML', z_min_bc='Metal', z_max_bc='Metal'):
        """
        Adds a 2.5D FDTD (varFDTD) solver region.
        """
        self.lum.addvarfdtd()

        # General
        self.lum.set('simulation time', simulation_time)

        # Geometry
        self._draw_3D_box(
            x=x, x_span=x_span, x_min=x_min, x_max=x_max, 
            y=y, y_span=y_span, y_min=y_min, y_max=y_max, 
            z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        
        # Effective index
        self.lum.set('x0', x0)
        self.lum.set('y0', y0)
        self.lum.set('effective index method', index_method)
        self.lum.set('polarization', polarization)
        
        # Mesh settings
        self.lum.set('mesh accuracy', mesh_accuracy)

        # Boundary conditions
        self.lum.set('x min bc', x_min_bc)
        self.lum.set('x max bc', x_max_bc)
        self.lum.set('y min bc', y_min_bc)
        self.lum.set('y max bc', y_max_bc)
        self.lum.set('z min bc', z_min_bc)
        self.lum.set('z max bc', z_max_bc)

    def add_eme_3D(self, x_min:float, wl:float, group_spans:np.ndarray, num_cells:np.ndarray, subcell_methods:np.ndarray, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, num_modes=10, temperature=300, y_min_bc='Metal', y_max_bc='Metal', z_min_bc='Metal', z_max_bc='Metal'):
        """
        Adds an Eigenmode Expansion (EME) solver region in the simulation.

        Parameters
        ----------
        wl : float
            Simulation wavelength in meters.
        group_spans : ndarray
            Array of group spans along the x-direction.
        num_cells : ndarray
            Array representing the number of cells in each group.
        subcell_methods : ndarray
            Array representing whether each group uses CVCS method.
        temperature : float, optional
            Simulation temperature in Kelvin.
        """
        self.lum.addeme()
        self.lum.set('simulation temperature', temperature)
        self.lum.set('solver type', '3D: X Prop')
        self.lum.set('wavelength', wl)

        # Cell group definition
        self.lum.set('number of cell groups', len(group_spans))
        self.lum.set('number of modes for all cell groups', num_modes)
        self.lum.set('group spans', group_spans)
        self.lum.set('cells', num_cells)
        self.lum.set('subcell method', subcell_methods)

        # EME region
        self.lum.set('x min', x_min)
        self._set_y_coords(y=y, y_span=y_span, y_min=y_min, y_max=y_max)
        self._set_z_coords(z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        
        self.lum.set('display cells', True)

        # Boundary conditions
        self.lum.set('y min bc', y_min_bc)
        self.lum.set('y max bc', y_max_bc)
        self.lum.set('z min bc', z_min_bc)
        self.lum.set('z max bc', z_max_bc)

    def create_beam(self, radius:float, direction='2D X normal', distance=0.0):
        """
        Creates a gaussian beam profile.

        Parameters
        ----------
        radius : float
            Waist radius in meters.
        direction : str, optional
            Beam direction. Either "2D X normal", "2D Y normal", or "2D Z normal".
        distance : float, optional
            Distance from waist in meters.
        """
        self.lum.setanalysis('use fully vectorial thin lens beam profile', False)
        
        self.lum.setanalysis('beam direction', direction)
        self.lum.setanalysis('define gaussian beam by', 'waist size and position')
        self.lum.setanalysis('waist radius', radius)
        self.lum.setanalysis('distance from waist', distance)
        
        self.lum.createbeam()

    def add_mode_source(self, center_wl:float, wl_span:float, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, axis='x-axis', direction='forward'):
        """
        Adds a mode source in the simulation. Works for varFDTD.

        Parameters
        ----------
        center_wl : float
            Center wavelength in meters.
        wl_span : float
            Wavelength span in meters.
        axis : str
            Propagation axis. May be either "x-axis", "y-axis", or "z_axis".
        direction : str, optional
            Either "forward" or "backward".
        """
        self.lum.addmodesource()

        # General
        self.lum.set('injection axis', axis)
        self.lum.set('direction', direction)

        # Geometry
        if axis == 'x-axis':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=None, z_span=None, z_min=None, z_max=None)
        elif axis == 'y-axis':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=None, z_span=None, z_min=None, z_max=None)
        else:
            raise ValueError(f'Input parameter <axis> must be either "x-axis" or "y-axis". It was given "{axis}".')

        # Frequency/Wavelength
        self.lum.set('set wavelength', True)
        self.lum.set('center wavelength', center_wl)
        self.lum.set('wavelength span', wl_span)


    def add_eme_profile_monitor(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D X-normal', name='Profile Monitor'):
        """
        Adds a profile monitor for an EME solver region.
        """
        self.lum.addemeprofile()

        # General
        self.lum.set('name', name)

        # Geometry
        self.lum.set('monitor type', monitor_type)

        if monitor_type == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{monitor_type}".')
        
    def add_power_monitor(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D X-normal', name='Power Monitor'):
        """
        Adds a field and power monitor in the simulation.

        Parameters
        ----------
        monitor_type : str, optional
            Monitor orientation.
        name : str, optional
            Monitor name.
        """
        self.lum.addpower()

        # General
        self.lum.set('name', name)

        # Geometry
        self.lum.set('monitor type', monitor_type)

        if monitor_type == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{monitor_type}".')
        

class LumericalHEAT(LumericalBase):
    """
    Class for HEAT simulations.
    """
    def __init__(self, lum) -> None:
        super().__init__(lum)

class LumericalCHARGE(LumericalBase):
    """
    Class for CHARGE simulations.
    """
    def __init__(self, lum) -> None:
        super().__init__(lum)

    def add_3D_charge(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, name='CHARGE Region'):
        """
        Adds a 3D CHARGE solver region.
        """
        self.lum.select('simulation region')

        # General
        self.lum.set('name', name)
        self.lum.set('dimension', '3D')

        # Geometry
        self._draw_3D_box(
            x=x, x_span=x_span, x_min=x_min, x_max=x_max,
            y=y, y_span=y_span, y_min=y_min, y_max=y_max,
            z=z, z_span=z_span, z_min=z_min, z_max=z_max)
    
    def add_2d_charge(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, dimension='2D X-Normal', name='CHARGE Region'):
        """
        Adds a 2D CHARGE solver region.
        """
        self.lum.select('simulation region')

        # General
        self.lum.set('name', name)
        self.lum.set('dimension', dimension)

        # Geometry
        if dimension == '2D X-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif dimension == '2D Y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif dimension == '2D Z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D X-normal", "2D Y-normal", or "2D Z-normal". It was given "{dimension}".')
        
    def add_charge_solver(self, solver_mode='steady state', temperature_dependence='isothermal', temperature=300):
        """
        Adds a CHARGE solver. Required for CHARGE monitors to be added.

        Parameters
        ----------
        solver_mode : str, optional
        temperature_dependence : str, optional
        temperature : float, optional
        """
        self.lum.addchargesolver()

        # General
        self.lum.set('solver mode', solver_mode)
        self.lum.set('temperature dependence', temperature_dependence)
        self.lum.set('simulation temperature', temperature)

    def add_charge_monitor(self, x=None, x_span=None, x_min=None, x_max=None, y=None, y_span=None, y_min=None, y_max=None, z=None, z_span=None, z_min=None, z_max=None, monitor_type='2D x-normal', name='Charge Monitor'):
        """
        Adds a CHARGE monitor.
        """
        self.lum.addchargemonitor()

        # General
        self.lum.set('name', name)
        self.lum.set('monitor type', monitor_type)

        # Geometry
        if monitor_type == '2D x-normal':
            self._draw_2D_box_x(
                x=x,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D y-normal':
            self._draw_2D_box_y(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y,
                z=z, z_span=z_span, z_min=z_min, z_max=z_max)
        elif monitor_type == '2D z-normal':
            self._draw_2D_box_z(
                x=x, x_span=x_span, x_min=x_min, x_max=x_max,
                y=y, y_span=y_span, y_min=y_min, y_max=y_max,
                z=z)
        else:
            raise ValueError(f'Input parameter <monitor_type> must be either "2D x-normal", "2D y-normal", or "2D z-normal". It was given "{monitor_type}".')
        
    def get_capacitance(self, monitor_name:str):
        """
        Returns the capacitance.
        """
        charge = self.lum.getresult(monitor_name, 'total_charge')

        Qn = electron_volt * charge['n']
        Qp = electron_volt * charge['p']
        V = charge['V_drain']
        L = len(V)

        Cn = (Qn[1:L] - Qn[0:L-1]) / (V[1] - V[0])
        Cp = (Qp[1:L] - Qp[0:L-1]) / (V[1] - V[0])
        Vmid = (V[0:L-1] + V[1:L]) / 2

        return Vmid, Cn, Cp