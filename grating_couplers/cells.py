import numpy as np
from scipy.constants import c, mu_0

class bloch_cell():
    def __init__(self):
        self.N = None
        self.dx = None
        self.dy = None
        self.domain_size = None
        self.x_coords = None
        self.y_coords = None

        self.sim_opts = None
        self.k = None
        self.Phi = None
        self.num_cells = None
        self.E_z = None
        self.k_vs_mode = None
        self.Phi_vs_mode = None
        self.chosen_mode_num = None
        self.E_z_for_overlap = None
        self.background_index = None

        self.R_est = None

class grating_cell(bloch_cell):
    def __init__(self):
        super().__init__()

        self.wg_min_y = None
        self.wg_max_y = None

        self.directivity = None
        self.max_angle_up = None
        self.max_angle_down = None
        self.P_rad_down = None
        self.P_rad_up = None
        self.P_in = None
        self.alpha_up = None
        self.alpha_down = None
        self.Sx = None
        self.P_per_x_slice = None
        self.Sy = None
        self.P_per_y_slice = None 
        self.Sy_up = None
        self.P_thru = None
        self.alpha_up_from_srad = None
        self.n_top = None
        self.n_bot = None