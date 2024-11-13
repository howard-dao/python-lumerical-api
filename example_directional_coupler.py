import sys
sys.path.append(r'C:\Program Files\Lumerical\v231\api\python')
import lumapi

import numpy as np
from utilities import LumericalFDTD
from directional_couplers import circular_directional_coupler


def generate_directional_coupler():
    fdtd.add_rect(
        name='Input Port',
        x_min=-coupling_length/2-bend_length-4e-6,
        x_max=-coupling_length/2-bend_length,
        y_min=gap/2+span,
        y_max=gap/2+span+width,
        z=0,
        z_span=t_Si,
        material=silicon)
    fdtd.add_rect(
        name='Add Port', 
        x_min=-coupling_length/2-bend_length-4e-6,
        x_max=-coupling_length/2-bend_length,
        y_min=-gap/2-span-width,
        y_max=-gap/2-span,
        z=0,
        z_span=t_Si,
        material=silicon)
    
    fdtd.add_poly(
        name='Input-Through',
        x=-coupling_length/2-bend_length,
        y=gap/2+span+width/2,
        z=0,
        z_span=t_Si,
        vertices=v1,
        material=silicon,
        alpha=1)
    fdtd.add_poly(
        name='Add-Cross',
        x=-coupling_length/2-bend_length,
        y=gap/2+span-width/2,
        z=0,
        z_span=t_Si,
        vertices=v2,
        material=silicon,
        alpha=1)
    
    fdtd.add_rect(
        name='Thru Port', 
        x_min=dc_length/2, 
        x_max=dc_length/2+4e-6, 
        y_min=gap/2+span, 
        y_max=gap/2+span+width, 
        z=0,
        z_span=t_Si, 
        material=silicon)
    fdtd.add_rect(
        name='Cross Port',
        x_min=dc_length/2,
        x_max=dc_length/2+4e-6,
        y_min=-gap/2-span-width,
        y_max=-gap/2-span,
        z=0,
        z_span=t_Si,
        material=silicon)
    
def place_simulation():
    # FDTD region settings
    z_span_fdtd = 4.0e-6
    fdtd_offset = 0.2e-6

    fdtd.add_fdtd_3D(
        x=0,
        x_span=dc_length+2*fdtd_offset,
        y=0,
        y_span=gap+width+2*span+2*fdtd_offset+z_span_fdtd,
        z=0,
        z_span=z_span_fdtd,
        mesh_accuracy=mesh_accuracy,
        background_material=oxide,
        simulation_time=simulation_time)
    
def place_monitors():
    # Monitor settings
    monitor_span = 4.0e-6
    monitor_offset = 0.1e-6
    x_monitor = dc_length/2 + monitor_offset
    y_monitor = gap/2 + width/2 + span

    fdtd.add_power_monitor(
        name='Thru Power Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=y_monitor,
        y_span=monitor_span,
        z=0, z_span=monitor_span)
    fdtd.add_power_monitor(
        name='Cross Power Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=-y_monitor,
        y_span=monitor_span,
        z=0,
        z_span=monitor_span)
    
    fdtd.add_expansion_monitor(
        name='Thru Expansion Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=y_monitor,
        y_span=monitor_span,
        z=0,
        z_span=monitor_span,
        center_wl=lam0,
        wl_span=wl_span)
    fdtd.add_expansion_monitor(
        name='Cross Expansion Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=-y_monitor,
        y_span=monitor_span,
        z=0,
        z_span=monitor_span,
        center_wl=lam0,
        wl_span=wl_span)
    
    fdtd.set_expansion(
        expansion_monitor='Thru Expansion Monitor',
        power_monitor='Thru Power Monitor',
        port='port')
    fdtd.set_expansion(
        expansion_monitor='Cross Expansion Monitor',
        power_monitor='Cross Power Monitor',
        port='port')
    
def place_source():
    # Source settings
    source_span = 4.0e-6
    source_offset = 0.1e-6
    x_source = -dc_length/2 - source_offset
    y_source = gap/2 + width/2 + span

    fdtd.add_mode(
        center_wl=lam0,
        wl_span=wl_span,
        x=x_source,
        y=y_source,
        y_span=source_span,
        z=0,
        z_span=source_span,
        axis='x-axis',
        direction='forward')

if __name__ == '__main__':

    # Materials
    silicon = 'Si (Silicon) - Palik'
    oxide = 'SiO2 (Glass) - Palik'

    # Waveguide thickness [m]
    t_Si    = 0.220e-6

    # Directional coupler parameters [m]
    radius  = 10.0e-6
    width   = 0.5e-6
    span    = 3.0e-6
    gap     = 0.2e-6
    coupling_length = 10e-6
    num_pts = 200

    # Source settings
    lam0    = 1.550e-6
    wl_span = 0.200e-6

    # Simulation settings
    mesh_accuracy = 2
    simulation_time = 2000e-15

    # Circular bend vertices
    v1, v2, bend_length = circular_directional_coupler(
        width=width,
        radius=radius,
        span=span,
        coupling_length=coupling_length,
        gap=gap,
        num_pts=num_pts)
    
    # Directional coupler total length
    dc_length = 2*bend_length + coupling_length

    # Initialize FDTD simulation
    fdtd = lumapi.FDTD(hide=False)
    fdtd = LumericalFDTD(lum=fdtd)

    # Generate directional coupler
    generate_directional_coupler()
    place_simulation()
    place_monitors()
    place_source()
    print('Simulation setup complete, ready to run.')