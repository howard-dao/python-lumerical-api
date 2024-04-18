import sys
sys.path.append(r'C:\Program Files\Lumerical\v231\api\python')
import lumapi

import numpy as np
from utilities import *
from geometry import circular_s_bend


def generate_directional_coupler():
    fdtd.add_rect(
        name='Input Port', 
        x_min=-coupling_length/2-bend_length-4e-6, x_max=-coupling_length/2-bend_length,
        y_min=gap/2+span, y_max=gap/2+span+width,
        z_min=z_min_Si, z_max=z_max_Si,
        material=silicon)
    fdtd.add_rect(
        name='Add Port', 
        x_min=-coupling_length/2-bend_length-4e-6, x_max=-coupling_length/2-bend_length,
        y_min=-gap/2-span-width, y_max=-gap/2-span,
        z_min=z_min_Si, z_max=z_max_Si,
        material=silicon)
    
    fdtd.add_poly(
        name='Input Bend', 
        x=-coupling_length/2, y=gap/2+width/2, z_min=z_min_Si, z_max=z_max_Si, points=verts_input, material=silicon)
    fdtd.add_poly(
        name='Add Bend', 
        x=-coupling_length/2, y=-gap/2-width/2, z_min=z_min_Si, z_max=z_max_Si, points=verts_add, material=silicon)
    
    fdtd.add_rect(
        name='Coupling Waveguide 1', 
        x=0, x_span=coupling_length, 
        y_min=gap/2, y_max=gap/2+width, 
        z_min=z_min_Si, z_max=z_max_Si, 
        material=silicon)
    fdtd.add_rect(
        name='Coupling Waveguide 2', 
        x=0, x_span=coupling_length, 
        y_min=-gap/2-width, y_max=-gap/2, 
        z_min=z_min_Si, z_max=z_max_Si, 
        material=silicon)

    fdtd.add_poly(
        name='Thru Bend', 
        x=coupling_length/2, 
        y=gap/2+width/2, 
        z_min=z_min_Si, z_max=z_max_Si, 
        points=verts_thru, material=silicon)
    fdtd.add_poly(
        name='Cross Bend', 
        x=coupling_length/2, 
        y=-gap/2-width/2, 
        z_min=z_min_Si, z_max=z_max_Si, 
        points=verts_cross, material=silicon)
    
    fdtd.add_rect(
        name='Thru Port', 
        x_min=coupling_length/2+bend_length, x_max=coupling_length/2+bend_length+4e-6, 
        y_min=gap/2+span, y_max=gap/2+span+width, 
        z_min=z_min_Si, z_max=z_max_Si, 
        material=silicon)
    fdtd.add_rect(
        name='Cross Port', 
        x_min=coupling_length/2+bend_length, x_max=coupling_length/2+bend_length+4e-6, 
        y_min=-gap/2-span-width, y_max=-gap/2-span, 
        z_min=z_min_Si, z_max=z_max_Si, 
        material=silicon)
    
def generate_oxide():
    # Oxide layers
    z_min_tox   = 0
    z_max_tox   = z_min_tox + t_tox
    z_max_box   = z_min_tox
    z_min_box   = z_max_box - t_box
    
    fdtd.add_rect(
        name='BOX',
        x=0, x_span=coupling_length+2*bend_length+12e-6,
        y=0, y_span=gap/2+width+span+10e-6,
        z_min=z_min_box, z_max=z_max_box,
        material=oxide,
        mesh_order=3,
        alpha=1)
    fdtd.add_rect(
        name='TOX',
        x=0, x_span=coupling_length+2*bend_length+12e-6,
        y=0, y_span=gap/2+width+span+10e-6,
        z_min=z_min_tox, z_max=z_max_tox,
        material=oxide,
        mesh_order=3,
        alpha=0.3)

def place_monitors():
    # Monitor settings
    monitor_offset = 0.1e-6
    monitor_span = 4e-6
    x_monitor = coupling_length/2 + bend_length + monitor_offset
    y_monitor = gap/2 + width/2 + span

    fdtd.add_power_monitor(
        name='Thru Power Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=y_monitor, y_span=monitor_span,
        z=z_mid_Si, z_span=monitor_span)
    fdtd.add_power_monitor(
        name='Cross Power Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=-y_monitor, y_span=monitor_span,
        z=z_mid_Si, z_span=monitor_span)
    
    fdtd.add_expansion_monitor(
        name='Thru Expansion Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=y_monitor, y_span=monitor_span,
        z=z_mid_Si, z_span=monitor_span,
        center_wl=lam0, wl_span=wl_span)
    fdtd.add_expansion_monitor(
        name='Cross Expansion Monitor',
        monitor_type='2D X-normal',
        x=x_monitor,
        y=-y_monitor, y_span=monitor_span,
        z=z_mid_Si, z_span=monitor_span,
        center_wl=lam0, wl_span=wl_span)
    
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
    source_offset = 0.1e-6
    x_source = -coupling_length/2 - bend_length - source_offset
    y_source = gap/2 + width/2 + span

    fdtd.add_mode_source(
        center_wl=lam0, wl_span=wl_span,
        x=x_source,
        y=y_source, y_span=4e-6,
        z=z_mid_Si, z_span=4e-6,
        axis='x-axis', direction='forward')

def place_simulation():
    # FDTD region settings
    z_span_fdtd = 4e-6
    fdtd_offset = 0.2e-6

    fdtd.add_fdtd_3D(
        x=0, x_span=coupling_length+2*bend_length+2*fdtd_offset,
        y=0, y_span=gap+width+2*span+2*fdtd_offset+z_span_fdtd,
        z=z_mid_Si, z_span=z_span_fdtd,
        mesh_accuracy=1, simulation_time=2000e-15)

if __name__ == '__main__':

    # Layer thicknesses [m]
    t_box   = 6.000e-6
    t_Si    = 0.220e-6
    t_tox   = 6.000e-6

    z_min_Si    = 0
    z_max_Si    = z_min_Si + t_Si
    z_mid_Si    = (z_min_Si + z_max_Si) / 2

    # Directional coupler parameters [m]
    radius  = 10.0e-6
    width   = 0.5e-6
    span    = 3.0e-6
    gap     = 0.2e-6
    coupling_length = 10e-6
    silicon = 'Si (Silicon) - Palik'
    oxide = 'SiO2 (Glass) - Palik'

    # Circular bend vertices
    verts_input = circular_s_bend(
        width=width, radius=radius, span=span, reflect=False, angle=180, n_points=100)
    verts_add = circular_s_bend(
        width=width, radius=radius, span=span, reflect=True, angle=180, n_points=100)
    verts_thru = circular_s_bend(
        width=width, radius=radius, span=span, reflect=True, angle=0, n_points=100)
    verts_cross = circular_s_bend(
        width=width, radius=radius, span=span, reflect=False, angle=0, n_points=100)
    
    # Bend length (derived from vertices)
    bend_length = np.max(verts_cross[:,0]) - verts_cross[0,0]

    # Source settings
    lam0    = 1.550e-6
    wl_span = 0.200e-6

    # Initialize FDTD simulation
    fdtd = lumapi.FDTD(hide=False)
    fdtd = LumericalFDTD(lum=fdtd)

    # Generate directional coupler
    print('Generating directional coupler...')
    generate_directional_coupler()
    generate_oxide()
    place_monitors()
    place_source()
    place_simulation()

    run = input('Run simulation (Y/N)?')
    if run == 'Y':
        fdtd.lum.run()
    else:
        print('Not simulating.')