import numpy as np
from scipy.sparse import spdiags, eye, coo_matrix
from scipy.sparse.linalg import eigs

def complexk_mode_solver_2D_PML(N:np.ndarray[float], disc:float, k0, num_modes:int, guess_k, BC:int, PML_options):

    # Total length of unwrapped vectors
    (ny, nx) = N.shape
    n_elem = nx * ny

    # Relative Permittivity
    er = N ** 2

    if PML_options[0] == 1:

        # Grab parameters
        pml_len = PML_options[1]
        pml_str = PML_options[2]
        pml_order = PML_options[3]

        # Setup discretizations
        ny_pml = 2 * round(pml_len / disc)
        if abs(ny_pml - round(ny_pml)) >= 1e-5:
            # Discretization was not integer value
            raise ValueError('Integer number of discretizations did not fit into the PML.')
        y_indx = range(ny_pml)

        # Using polynomial strength PML
        pml_y = (1 + 1j*pml_str*(y_indx / ny_pml) ** (pml_order))
        pml_y = np.conj(pml_y)

        # Draw stretched coordinate PML
        pml_y_all:np.ndarray = np.ones(2*ny, ny)
        pml_y_all[0:ny_pml-1, :] = np.tile(np.flipud(pml_y[0:-2]), (1, nx))
        pml_y_all[-ny_pml,:] = np.tile(pml_y, (1, nx))

        # Stretched coordinate operator
        pml_y_all_vec = np.reshape(pml_y_all.T, (pml_y_all.size, 1))
        Sy_f = spdiags(1/pml_y_all_vec[1:-1:2], 0, n_elem, n_elem)
        Sy_b = spdiags(1/pml_y_all[0:-2:2], 0, n_elem, n_elem)

    # Generate forward Dy
    diag0 = -np.ones((n_elem, 1))
    diag1 = np.ones((n_elem, 1))
    diag1[ny:-1:ny] = 0
    if BC == 1:
        diag0[ny:-1:ny] = 0
    
    diag1[1:-1] = diag1[0:-2]
    diag_all = np.hstack((diag0, diag1))
    diag_indexs = [0, 1]
    Dy_f = (1/disc) * spdiags(diag_all, diag_indexs, n_elem, n_elem)

    if PML_options[0] == 1:
        Dy_f = Sy_f * Dy_f

    # Generate backward Dy
    diag0 = np.ones((n_elem, 1))
    diagm1 = -np.ones((n_elem, 1))
    diagm1[0:-1:ny] = 0
    if BC == 1:
        diag0[0:-1:ny] = 0

    # Stitch together the diags
    diag_all = np.hstack((diag1, diag0))
    diag_indexs = [-1, 0]

    Dy_b = (1/disc) * spdiags(diag_all, diag_indexs, n_elem, n_elem)

    if PML_options[0] == 1:
        Dy_b = Sy_b * Dy_b

    # Generate Dy squared
    Dy2 = Dy_b * Dy_f


    # Generate forward Dx
    diag0 = np.ones((n_elem, 1))
    diagP = np.ones((n_elem, 1))
    diagBC = np.ones((n_elem, 1))
    diag_all = np.hstack((diagBC, diag0, diagP))
    diag_indexes = [ny-n_elem, 0, ny]

    Dx_f = (1/disc) * spdiags(diag_all, diag_indexes, n_elem, n_elem)

    # Generate backward Dy
    diag0 = np.ones((n_elem, 1))
    diagM = -np.ones((n_elem, 1))
    diagBC = -np.ones((n_elem, 1))
    diag_all = np.hstack((diagM, diag0, diagBC))
    diag_indexes = [-ny, 0, (n_elem-ny)]

    Dx_b = (1/disc) * spdiags(diag_all, diag_indexes, n_elem, n_elem)

    # Generate Dx squared
    Dx2 = Dx_b * Dx_f

    # Make Eigenvalue Equation
    # n2 = spdiags()
    A = Dx2 + Dy2 + (k0**2)*n2
    B = 1j * (Dx_b + Dx_f)
    I = eye(n_elem, n_elem)
    Z = coo_matrix((n_elem, n_elem))
    LH = np.vstack((np.hstack((A,B)), np.hstack((Z,I))))
    RH = np.vstack((np.hstack((Z,I)), np.hstack((I,Z))))

    (Phi_all, k_all) = eigs(LH, RH, num_modes, guess_k)
    k_all = np.diag(k_all)

    # Phi_all = Phi_all
    Phi_all = np.reshape(Phi_all, ny, nx, Phi_all.shape[1])

    return