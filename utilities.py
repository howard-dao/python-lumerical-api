import numpy as np
from os import getcwd, listdir, remove
from os.path import join
from shutil import move


class LumericalBase():
    def __init__(self, lum) -> None:
        self.lum = lum

    def save_file_in_dir(self, file:str, path=getcwd(), overwrite=False):
        """
        Saves a file to a specified folder. If a file with the same name already exists,
        one may choose to overwrite it.

        Parameters:
            file
                type: str
                desc: name of the file
            path
                type: str
                desc: name of the destination directory, current directory by default
            overwrite
                type: bool
                desc: whether or not to overwrite an existing file with the same name
        """
        files = listdir(path)
        if file in files and overwrite:
            print('File already exists, overwriting...')
            remove(join(path, file))
        try:
            move(file, path)
        except:
            print('File already exists, not overwriting')

        # if file in files:
        #     if overwrite:
        #         print('File already exists, overwriting...')
        #         remove(join(path, file))
        #         move(file, path)
        #     else:
        #         print('File already exists, not overwriting')
        # else:
        #     move(file, path)
            

    def add_rect(self, x_min:float, x_max:float, y_min:float, y_max:float, z_min:float, z_max:float, 
                 material, mesh_order=2, name='rectangle', alpha=1.0):
        """
        Adds a rectangle object in the simulation.
        """
        self.lum.addrect()
        self.lum.set('name', name)
        self.lum.set('x min', x_min)
        self.lum.set('x max', x_max)
        self.lum.set('y min', y_min)
        self.lum.set('y max', y_max)
        self.lum.set('z min', z_min)
        self.lum.set('z max', z_max)

        # Set the material.
        # If <material> is a string, this sets the object material to a predefined material in Lumerical database.
        # This will fail if <material> is not in the database.
        # If <material> is a float, the object's refractive index is set instead.
        if isinstance(material, str):
            self.lum.set('material', material)
        elif isinstance(material, float):
            self.lum.set('index', material)

        # Set the alpha. Affects only the object's appearance in the simulation.
        self.lum.set('alpha', alpha)

        # Set mesh order. By default, it is 2.
        self.lum.set('override mesh order from material database', 1)
        self.lum.set('mesh order', mesh_order)

    def add_circle(self, x:float, y:float, z_min:float, z_max:float, 
                   radius:float, axis, theta:float, material, mesh_order=2, name='circle', alpha=1.0):
        """
        Adds a cylinder object in the simulation.
        """
        self.lum.addcircle()
        self.lum.set('name', name)
        self.lum.set('x', x)
        self.lum.set('y', y)
        self.lum.set('z min', z_min)
        self.lum.set('z max', z_max)
        self.lum.set('radius', radius)
        self.lum.set('first axis', axis)
        self.lum.set('rotation 1', theta)
        if isinstance(material, str):
            self.lum.set('material', material)
        elif isinstance(material, float):
            self.lum.set('index', material)
        self.lum.set('alpha', alpha)
        self.lum.set('override mesh order from material database', 1)
        self.lum.set('mesh order', mesh_order)

    def add_poly(self, x:float, y:float, z_min:float, z_max:float, 
                 points, material, mesh_order=2, name='polygon', alpha=1.0):
        """
        Adds a polygon object in the simulation.
        """
        self.lum.addpoly()
        self.lum.set('vertices', points)
        self.lum.set('name', name)
        self.lum.set('x', x)
        self.lum.set('y', y)
        self.lum.set('z min', z_min)
        self.lum.set('z max', z_max)
        if isinstance(material, str):
            self.lum.set('material', material)
        elif isinstance(material, float):
            self.lum.set('index', material)
        self.lum.set('alpha', alpha)
        self.lum.set('override mesh order from material database', 1)
        self.lum.set('mesh order', mesh_order)

    def add_mesh(self, x_min:float, x_max:float, y_min:float, y_max:float, z_min:float, z_max:float, 
                 dx=None, dy=None, dz=None, structure=None, name='Mesh'):
        """
        Adds a mesh override to a specific area or structure in the simulation.
        """
        self.lum.addmesh()
        self.lum.set('name', name)

        # Geometry settings
        self.lum.set('x min', x_min)
        self.lum.set('x max', x_max)
        self.lum.set('y min', y_min)
        self.lum.set('y max', y_max)
        self.lum.set('z min', z_min)
        self.lum.set('z max', z_max)

        # If <structure> is specified as a string, override the mesh of the structure.
        # This only takes effect if <structure> is exactly equal to the name of the object
        # that's being overridden, not the name of the mesh itself.
        if isinstance(structure, str):
            self.lum.set('based on a structure', 1)
            self.lum.set('structure', structure)

        # Sets the mesh overrides if they are specified
        if dx is not None:
            self.lum.set('override x mesh', 1)
            self.lum.set('dx', dx)
        if dy is not None:
            self.lum.set('override y mesh', 1)
            self.lum.set('dy', dy)
        if dz is not None:
            self.lum.set('override z mesh', 1)
            self.lum.set('dz', dz)



    def add_index_monitor(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float, 
                          monitor_type='2D Z-normal', name='index monitor'):
        """
        Adds an index monitor.
        """
        self.lum.addindex()
        self.lum.set('name', name)
        self.lum.set('monitor type', monitor_type)
        if monitor_type == '2D X-normal':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif monitor_type == '2D Y-normal':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        else:
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)

class LumericalFDTD(LumericalBase):
    def __init__(self, fdtd) -> None:
        super().__init__(lum=fdtd)

    def add_fdtd_3D(self, x_min:float, x_max:float, y_min:float, y_max:float, z_min:float, z_max:float, 
                    x_min_bc='PML', x_max_bc='PML', y_min_bc='PML', y_max_bc='PML', z_min_bc='PML', z_max_bc='PML',
                    mesh_accuracy=2, simulation_time=1000e-15):
        """
        Adds a 3D FDTD simulation region.
        """
        self.lum.addfdtd()

        # General Settings
        self.lum.set('dimension', '3D')
        self.lum.set('simulation time', simulation_time)

        # Geometry
        self.lum.set('x min', x_min)
        self.lum.set('x max', x_max)
        self.lum.set('y min', y_min)
        self.lum.set('y max', y_max)
        self.lum.set('z min', z_min)
        self.lum.set('z max', z_max)

        # Boundary Conditions
        self.lum.set('x min bc', x_min_bc)
        self.lum.set('x max bc', x_max_bc)
        self.lum.set('y min bc', y_min_bc)
        self.lum.set('y max bc', y_max_bc)
        self.lum.set('z min bc', z_min_bc)
        self.lum.set('z max bc', z_max_bc)

        # Mesh Settings
        self.lum.set('mesh accuracy', mesh_accuracy)

    def set_global_monitors(self, center_wl:float, wl_span:float, use_wl_spacing=True, n_points=21):
        """
        Set global monitor settings.
        """
        if use_wl_spacing:
            self.lum.setglobalmonitor('use wavelength spacing', 1)
        else:
            self.lum.setglobalmonitor('use wavelength spacing', 0)
            
        self.lum.setglobalmonitor('wavelength center', center_wl)
        self.lum.setglobalmonitor('wavelength span', wl_span)
        self.lum.setglobalmonitor('frequency points', n_points)

    def add_power_monitor(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float, 
                          monitor_type='2D Z-normal', name='power monitor'):
        """
        Adds a field and power monitor in the simulation.
        """
        self.lum.addpower()
        self.lum.set('name', name)
        self.lum.set('monitor type', monitor_type)
        if monitor_type == '2D X-normal':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif monitor_type == '2D Y-normal':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        else:
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)

    def add_expansion_monitor(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float, 
                              center_wl:float, wl_span:float, monitor_type='2D X-normal', 
                              mode_selection='fundamental mode', n_points=21, name='expansion monitor'):
        """
        Adds a mode expansion monitor in the simulation.
        """
        self.lum.addmodeexpansion()
        self.lum.set('name', name)

        self.lum.set('monitor type', monitor_type)
        if monitor_type == '2D X-normal':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif monitor_type == '2D Y-normal':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        else:
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)

        if mode_selection == 'user select':
            self.lum.set('mode selection', mode_selection)
            self.lum.set('selected mode numbers', 1)
        
        self.lum.set('override global monitor settings', 1)
        self.lum.set('use wavelength spacing', 1)

        using_source = self.lum.get('use source limits')
        if not using_source:
            self.lum.set('center wavelength', center_wl)
            self.lum.set('wavelength span', wl_span)

        self.lum.set('frequency points', n_points)

    def set_expansion(self, expansion_monitor:str, power_monitor:str, port='Port'):
        """
        Associates a power monitor with a mode expansion monitor.

        Parameters:
            expansion_monitor
                type: str
                desc: name of the expansion monitor, must exist in the simulation
            power_monitor
                type: str
                desc: name of the DFT monitor, must exist in the simulation
            port
                type: str
                desc: name of expansion port from which data is collected
        """
        self.lum.select(expansion_monitor)
        self.lum.setexpansion(port, power_monitor)

    def add_movie_monitor(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float, 
                          monitor_type='2D Z-normal', name='movie monitor'):
        """
        Adds a movie monitor in the simulation.
        """
        self.lum.addmovie()
        self.lum.set('name', name)
        self.lum.set('monitor type', monitor_type)
        if monitor_type == '2D X-normal':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif monitor_type == '2D Y-normal':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        else:
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)

    def add_mode_source(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float, 
                        center_wl:float, wl_span:float, axis='x-axis', direction='forward'):
        """
        Adds a mode source in the simulation. Depending on the injection axis, 
        the span along that axis will be disabled.
        """
        self.fdtd.addmode()
        self.fdtd.set('injection axis', axis)
        self.fdtd.set('direction', direction)
        if axis == 'x-axis':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif axis == 'y-axis':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        else:
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
        self.fdtd.set('center wavelength', center_wl)
        self.fdtd.set('wavelength span', wl_span)

    def add_gauss_source(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float, 
                         radius:float, center_wl:float, wl_span:float, axis='x', direction='forward'):
        """
        Adds a Gaussian source in the simulation. Uses waist radius.
        """
        self.lum.addgaussian()
        self.lum.set('injection axis', axis)
        self.lum.set('direction', direction)
        if axis == 'x':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        elif axis == 'y':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
        else:
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            
        self.lum.set('center wavelength', center_wl)
        self.lum.set('wavelength span', wl_span)

        self.lum.set('use scalar approximation', 1)
        self.lum.set('waist radius w0', radius)
        self.lum.set('distance from waist', 0)
    
class LumericalMODE(LumericalBase):
    def __init__(self, mode) -> None:
        super().__init__(lum=mode)

    def add_fde(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float,
                x_min_bc='Metal', x_max_bc='Metal',
                y_min_bc='Metal', y_max_bc='Metal',
                z_min_bc='Metal', z_max_bc='Metal',
                solver_type='2D X normal'):
        """
        Adds a Finite Difference Eigenmode (FDE) solver region in the simulation
        """
        self.lum.addfde()
        self.lum.set('solver type', solver_type)
        if solver_type == '2D X normal':
            self.lum.set('x', x)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
            self.lum.set('y min bc', y_min_bc)
            self.lum.set('y max bc', y_max_bc)
            self.lum.set('z min bc', z_min_bc)
            self.lum.set('z max bc', z_max_bc)
        elif solver_type == '2D Y normal':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('z', z)
            self.lum.set('z span', z_span)
            self.lum.set('x min bc', x_min_bc)
            self.lum.set('x max bc', x_max_bc)
            self.lum.set('z min bc', z_min_bc)
            self.lum.set('z max bc', z_max_bc)
        elif solver_type == '2D Z normal':
            self.lum.set('x', x)
            self.lum.set('x span', x_span)
            self.lum.set('y', y)
            self.lum.set('y span', y_span)
            self.lum.set('z', z)
            self.lum.set('x min bc', x_min_bc)
            self.lum.set('x max bc', x_max_bc)
            self.lum.set('y min bc', y_min_bc)
            self.lum.set('y max bc', y_max_bc)

    def add_eme(self, x:float, x_span:float, y:float, y_span:float, z:float, z_span:float):
        """
        Adds an Eigenmode Expansion (EME) solver region in the simulation
        """
        self.lum.addeme()
        self.lum.set('x', x)
        self.lum.set('x span', x_span)
        self.lum.set('y', y)
        self.lum.set('y span', y_span)
        self.lum.set('z', z)
        self.lum.set('z span', z_span)