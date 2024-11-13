import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light as c
from scipy.constants import mu_0 as u0
from scipy.constants import epsilon_0 as e0
from scipy.sparse import spdiags, csr_matrix, lil_matrix

PEC = 0
PMCh = 1
PBC = 2
PMC = 3
PECh = 4

def sisolver3d(nlayers:np.ndarray, xlayers:np.ndarray, ylayers:np.ndarray, dxy:np.ndarray, k0:float, OPTS:dict):
    
    # Default values for OPTS dictionary
    if not 'enginever' in OPTS:
        OPTS['enginever'] = 'm2wcylR2'
    if not 'fieldmode' in OPTS:
        OPTS['fieldmode'] = 'V'     # V for Vectorial
    if not 'coordmode' in OPTS:
        OPTS['coordmode'] = 'C'     # C for Cartesian
    if not 'eigmode' in OPTS:
        OPTS['eigmode'] = 'b'       # b for beta propagation constant
    if not 'BC' in OPTS:
        OPTS['BC'] = np.zeros((1,4))
    if not 'PMLwidth' in OPTS:
        OPTS['PMLwidth'] = np.zeros((1,4))
    if not 'PMLsigma' in OPTS:
        OPTS['PMLsigma'] = np.ones((1,2))
        
    if 'mu_guess' in OPTS:
        mu_guess = OPTS['mu_guess']
    else:
        mu_guess = np.max(nlayers) * k0

    if 'NMODES_CALC' in OPTS:
        NMODES_CALC = OPTS['NMODES_CALC']
    else:
        NMODES_CALC = 1

    if 'NMODES_KEEP' in OPTS:
        NMODES_KEEP = OPTS['NMODES_KEEP']
    else:
        NMODES_KEEP = NMODES_CALC

    # Parse simpler boundary condition specification method
    if isinstance(OPTS['BC'], list):
        tmp = [None, None, None, None]
        for kk in range(4):
            match OPTS['BC'][kk]:
                case 'PEC':
                    tmp[kk] = 0
                case 'PMCh':
                    tmp[kk] = 1
                case 'PBC':
                    tmp[kk] = 2
                case 'PMC':
                    tmp[kk] = 3
                case 'PECh':
                    tmp[kk] = 4
                case _:
                    raise ValueError('Unrecognized boundary condition specified.')
        OPTS['BC'] = tmp

    # Parse the input structure
    MM = len(xlayers)
    NN = len(ylayers)
    dx = dxy[0]
    dy = dxy[1]

    xlayers[-1] = round(np.sum(xlayers)/dx)*dx - np.sum(xlayers[0:-1])
    ylayers[-1] = round(np.sum(ylayers)/dy)*dy - np.sum(ylayers[0:-1])

    # Interface (?)
    xint = np.insert(np.cumsum(xlayers), 0, 0)
    yint = np.insert(np.cumsum(ylayers), 0, 0)

    # Grid coordinates
    x = np.arange(0, xint[-1]+dx/2, dx/2)
    y = np.arange(0, yint[-1]+dy/2, dy/2)

    # Put edge interfaces to infinity to avoid index averaging at domain edges
    xint[0] = -np.inf
    xint[-1] = np.inf
    yint[0] = -np.inf
    yint[-1] = np.inf

    # Index averaging
    N = {}
    N['x'] = np.reshape(x, (len(x), 1))
    N['y'] = np.reshape(y, (len(y), 1))
    N['n'] = np.zeros((len(x), len(y)))

    xp = x[0:None:2]
    yp = y[0:None:2]
    npsqz = np.zeros((len(xp), len(yp)))

    for mm in range(MM):
        ixin = np.argwhere(((xp + dx/2) > xint[mm]) & ((xp - dx/2) <= xint[mm+1]))
        ixin0 = ixin[0][0]
        ixin1 = ixin[-1][0]

        xpmin = np.maximum(xint[mm], xp[ixin] - dx/2)
        xpmax = np.minimum(xint[mm+1], xp[ixin] + dx/2)

        for nn in range(NN):
            iyin = np.argwhere(((yp + dy/2) > yint[nn]) & ((yp - dy/2) <= yint[nn+1]))
            iyin0 = iyin[0][0]
            iyin1 = iyin[-1][0]

            ypmin = np.maximum(yint[nn], yp[iyin] - dy/2)
            ypmax = np.minimum(yint[nn+1], yp[iyin] + dy/2)

            npsqz[ixin0:ixin1+1, iyin0:iyin1+1] = npsqz[ixin0:ixin1+1, iyin0:iyin1+1] + (xpmax-xpmin)@np.transpose(ypmax-ypmin) / (dx*dy) * nlayers[nn,mm]**2

    xp = x[0:None:2]
    yp = y[1:-1:2]
    npsqy = np.Inf * np.ones((len(xp), len(yp)))
    npsqytot = np.zeros((len(xp), len(yp)))

    for mm in range(MM):
        ixin = np.argwhere(((xp + dx/2) > xint[mm]) & ((xp - dx/2) <= xint[mm+1]))
        ixin0 = ixin[0][0]
        ixin1 = ixin[-1][0]

        xpmin = np.maximum(xint[mm], xp[ixin] - dx/2)
        xpmax = np.minimum(xint[mm+1], xp[ixin] + dx/2)

        for nn in range(NN):
            iyin = np.argwhere(((yp + dy/2) > yint[nn]) & ((yp - dy/2) <= yint[nn+1]))
            iyin0 = iyin[0][0]
            iyin1 = iyin[-1][0]

            ypmin = np.maximum(yint[nn], yp[iyin] - dy/2)
            ypmax = np.minimum(yint[nn+1], yp[iyin] + dy/2)

            npsqy[ixin0:ixin1+1, iyin0:iyin1+1] = 1 / (1/npsqy[ixin0:ixin1+1, iyin0:iyin1+1] + 1 / ((xpmax-xpmin)@np.transpose(1/(ypmax-ypmin)) / (dx/dy) * nlayers[nn,mm]**2))

        npsqy[np.isinf(npsqy)] = 0
        npsqytot = npsqytot + npsqy
        npsqy = np.inf * np.ones((len(xp), len(yp)))

    xp = x[1:-1:2]
    yp = y[0:None:2]
    npsqx = np.inf * np.ones((len(xp), len(yp)))
    npsqxtot = np.zeros((len(xp), len(yp)))

    for nn in range(NN):
        iyin = np.argwhere(((yp + dy/2) > yint[nn]) & ((yp - dy/2) <= yint[nn+1]))
        iyin0 = iyin[0][0]
        iyin1 = iyin[-1][0]

        ypmin = np.maximum(yint[nn], yp[iyin] - dy/2)
        ypmax = np.minimum(yint[nn+1], yp[iyin] + dy/2)

        for mm in range(MM):
            ixin = np.argwhere(((xp + dx/2) > xint[mm]) & ((xp - dx/2) <= xint[mm+1]))
            ixin0 = ixin[0][0]
            ixin1 = ixin[-1][0]

            xpmin = np.maximum(xint[mm], xp[ixin] - dx/2)
            xpmax = np.minimum(xint[mm+1], xp[ixin] + dx/2)

            npsqx[ixin0:ixin1+1, iyin0:iyin1+1] = 1 / (1/npsqx[ixin0:ixin1+1, iyin0:iyin1+1] + 1/(1 / (xpmax-xpmin)@np.transpose(ypmax-ypmin) / (dy/dx) * nlayers[nn,mm]**2))
            
        npsqx[np.isinf(npsqx)] = 0
        npsqxtot = npsqxtot + npsqx
        npsqx = np.inf * np.ones((len(xp), len(yp)))

    N['n'][0:None:2, 0:None:2] = np.sqrt(npsqz)
    N['n'][0:None:2, 1:-1:2] = np.sqrt(npsqytot)
    N['n'][1:-1:2, 0:None:2] = np.sqrt(npsqxtot)

    N['n'][1:-1:2, 1:-1:2] = np.sqrt((N['n'][0:-2:2, 1:-1:2]**2 + N['n'][2:None:2, 1:-1:2]**2 + N['n'][1:-1:2, 0:-2:2]**2 + N['n'][1:-1:2, 2:None:2]**2) / 4)

    if np.argwhere(np.isinf(N['n'])):
        raise ValueError('Si solver error: Generated index distribution has non-finite elements.')
    
    # For bend modes
    if 'radius' in OPTS:
        R = OPTS['radius']
        N['x'] = N['x'] + R
        OPTS['coordmode'] = 'R'
    
    # # Call mode solver engine with prepared parameters
    # beta, F, V = eval(OPTS['enginever'], N, k0, mu_guess, OPTS, NMODES_CALC, OPTS['PMLwidth'], OPTS['PMLsigma'])
    
    # if OPTS['coordmode'] == 'R':
    #     alpha = beta.imag / R
    #     F['dB90'] = 20 * alpha * (np.pi*R/2) * np.log10(np.exp(1))
    # else:
    #     alpha = beta.imag

    # F['dBcm'] = 0.2 / np.log(10) * alpha
    # F['beta'] = beta
    # F['LossQ'] = beta.real / beta.imag / 2

    # # Remove undesired modes
    # F['Ex'] = F['Ex'][:,:,0:NMODES_KEEP]
    # F['Ey'] = F['Ey'][:,:,0:NMODES_KEEP]
    # F['Ez'] = F['Ez'][:,:,0:NMODES_KEEP]
    # F['Hx'] = F['Hx'][:,:,0:NMODES_KEEP]
    # F['Hy'] = F['Hy'][:,:,0:NMODES_KEEP]
    # F['Hz'] = F['Hz'][:,:,0:NMODES_KEEP]

    # F['dx'] = dx
    # F['dy'] = dy

    return N



def m2wcylR2(S:dict, nu_index:float, mu_guess:float, OPTS:dict, nmodes:int, dPML, sigmaPML):

    if not 'adjoint' in OPTS:
        OPTS['adjoint'] = 0
    if not 'eigsfcn' in OPTS:
        OPTS['eigsfcn'] = 'eigs'
    if not 'sigma' in OPTS:
        OPTS['sigma'] = []
    if not OPTS['sigma']:
        OPTS['sigma'] = mu_guess / mu_guess[0]
    if not 'BC' in OPTS:
        OPTS['BC'] = [0, 0, 0, 0]
    if not OPTS['BC']:
        OPTS['BC'] = [0, 0, 0, 0]
    BCl = OPTS['BC'][0]
    BCr = OPTS['BC'][1]
    BCd = OPTS['BC'][2]
    BCu = OPTS['BC'][3]
    if not 'operver' in OPTS:
        OPTS['operver'] = 'm2dpmloperR2'
    if not OPTS['operver']:
        OPTS['operver'] = 'm2dpmloperR2'

    R = S['x']
    Z = S['y']
    nn = S['n']

    L = nu_index

    sigmaRMAX = sigmaPML[0]
    sigmaZMAX = sigmaPML[1]

    NR = (len(R) + 1) / 2
    NZ = (len(Z) + 1) / 2

    if OPTS['eigmode'] == 'w':
        w0 = c * mu_guess
    else:
        w0 = c * nu_index

    # Generate PML matrices
    Pr, Pz, Rmtx = m2dpmlmatx(w=w0, er=nn**2, Rvec=R, Zvec=Z, sigmaMAX=[sigmaRMAX, sigmaZMAX], dPML=dPML)

    # Generate 'vector-Helmholtz' operator matrix
    dR = R[2] - R[0]
    dZ = Z[2] - Z[0]

    if OPTS['coordmode'] == 'C':
        Rmtx = np.ones_like(Rmtx)

    # H = eval(OPTS['operver'], nn**2, Pr, Pz, Rmtx, dR, dZ, L**2, OPTS)
    H = m2dpmloperR2(er=nn**2, Pr=Pr, Pz=Pz, rho=Rmtx, dr=dR, dz=dZ, keig=L**2, OPTS=OPTS)
    H = H / mu_guess**2

    oxL = 0
    oxR = 0
    oxD = (BCd == PEC | BCd == PECh)
    oxU = (BCu == PEC | BCu == PECh | BCu == PBC)

    oyL = (BCl == PEC | BCl == PECh)
    oyR = (BCr == PEC | BCr == PECh | BCr == PBC)
    oyD = 0
    oyU = 0

    MxSIZE = (NR - 1 - (oxL + oxR)) * (NZ - (oxD + oxU))
    MySIZE = (NR - (oyL + oyR)) * (NZ - 1 - (oyD + oyU))

    if OPTS['fieldmode'][0] == 'M':
        if OPTS['fieldmode'][1] == 'X':
            H = H[0:MxSIZE, 0:MxSIZE]
        elif OPTS['fieldmode'][1] == 'Y':
            H = H[MxSIZE + range(MySIZE), MxSIZE + range(MySIZE)]
        else:
            raise ValueError('.')
        
    if OPTS['adjoint']:
        H = H.T
    
    asgn = (-1)**OPTS['adjoint']

    V,D = eval(OPTS['eigsfcn'], H, nmodes, OPTS['sigma'], OPTS)

    mu = np.sqrt(np.diag(D)) * mu_guess[0]

    if OPTS['eigmode'] == 'w':
        X = np.sortrows([-mu.imag*asgn, mu.real, V.T])

        mu = -X[:,1] + 1j*X[:,0]*asgn
    else:
        X = np.sortrows(mu.imag*asgn, -mu.real, V.T)

        mu = -X[:,1] + 1j*X[:,0]*asgn

    V = X[:, 2:]
    D = np.diag((mu / mu_guess)**2)

    
    NX = NR
    NY = NZ

    F = {}
    F['Rr'] = R[2+2*oxL, 2*NR-2-2*oxR, 2]
    F['Zr'] = Z[1+2*oxD, 2*NZ-1-2*oxU, 2]
    F['Rz'] = R[1+2*oyL, 2*NR-1-2*oyR, 2]
    F['Zz'] = Z[2+2*oyD, 2*NZ-2-2*oyU, 2]

    er = nn**2
    Ex = np.zeros(NX-1-(oxL+oxR), NY-(oxD+oxU), len(V))
    Ey = np.zeros(NX-(oyL+oyR), NY-1-(oyD+oyU), len(V))
    betaR = np.zeros(1, len(V))
    omega = np.zeros(1, len(V))
    Gammah = np.zeros(1, len(V))
    Gammat = np.zeros(1, len(V))
    Omegat = np.zeros(1, len(V))

    # for k in range(len(V,2)):
    #     match OPTS['fieldmode']:
    #         case {'V': 'MX'}:
    #             # Ex[:,:,k] = np.reshape()
    #             pass
    #         case 'MY':
    #             pass
    #         case _:
    #             raise ValueError('.')
            
    #     # Set propagation constant along Z direction
    #     if OPTS['eigmode'] == 'b':
    #         betaR[k] = mu[k]
    #         omega[k] = w0
    #     else:
    #         betaR[k] = nu_index
    #         omega[k] = c * mu[k]

    #     jxx = np.arange(1+2*oxD, 2*NY-1-2*oxU, 2)
    #     iyy = np.arange(1+2*oyL, 2*NX-1-2*oyR, 2)
    #     erxx = er[1:2*NX-1, jxx]
    #     eryy = er[iyy, 2*NY-1, 2]
    #     erzz = er[iyy, jxx]
    #     rxx = Rmtx[1, 2*NX-1, jxx]
    #     ryx = Rmtx[iyy, jxx]

    #     Ax = rxx * erxx * Ex[:,:,k]
    #     Ax = Pr[iyy, jxx] * diffxy(Ax, 'Ex', [BCl, BCr, BCd, BCu], 0) / dR

    #     Ay = eryy * Ey[:,:,k]
    #     Ay = Pz[iyy, jxx] * ryx * diffxy(Ay, 'Ey', [BCl, BCr, BCd, BCu], 1) / dZ

    #     deltaphi = 0
    #     Gammah[k] = betaR[k] * np.exp(-1j * deltaphi)

    #     if Gammah[k] == 0:

    F['Ex'] = Ex
    F['Ey'] = Ey
    F['Ez'] = Ez
    F['Hx'] = Hx
    F['Hy'] = Hy
    F['Hz'] = Hz
    F['BC'] = [BCl, BCr, BCd, BCu]
    F['ver'] = 'R2'

    return mu, F, V, D, Pr, Pz, Rmtx



def m2dpmlmatx(w:float, er:np.ndarray, Rvec:np.ndarray, Zvec:np.ndarray, sigmaMAX:list, dPML:list):
    """
    PML matrix generator using complex coordinate stretching.
    """
    # PML thickness
    dPMLRlo = dPML[0]
    dPMLRhi = dPML[1]
    dPMLZlo = dPML[2]
    dPMLZhi = dPML[3]

    # Total and useful (without PML) computational domain edges
    RminT = np.min(Rvec)
    RmaxT = np.max(Rvec)
    ZminT = np.min(Zvec)
    ZmaxT = np.max(Zvec)

    Rmin = RminT + dPMLRlo
    Rmax = RmaxT - dPMLRhi
    Zmin = ZminT + dPMLZlo
    Zmax = ZmaxT - dPMLZhi

    # PML max conductivities for parabolic conductivity profile
    sigmaRMAX = sigmaMAX[0]
    sigmaZMAX = sigmaMAX[1]

    # Compex coordinate stretching (PML) factor
    fR = 1j * sigma(Rvec=Rvec, Rlimits=[Rmin,Rmax], sigmaMAX=sigmaRMAX) @ np.ones_like(Zvec.T) / (w*e0*er)
    fZ = 1j * np.ones_like(Rvec) @ np.transpose(sigma(Rvec=Zvec, Rlimits=[Zmin,Zmax], sigmaMAX=sigmaZMAX)) / (w*e0*er)

    # PML matrices
    Pr = 1 / (1 + fR)
    Pz = 1 / (1 + fZ)
    Rmtx = Rvec * np.ones_like(Zvec.T)
    dRmtx = rcomplex(Rvec=Rvec, Rlimits=[Rmin, Rmax], sigmaMAX=sigmaRMAX) * np.ones_like(Zvec.T)
    Rmtx = Rmtx + (dRmtx / (w*e0*er))

    return Pr, Pz, Rmtx

def sigma(Rvec:np.ndarray, Rlimits:list, sigmaMAX:float):
    """
    Parabolic conductivity function for PML layer.
    """
    Rmin = Rlimits[0]
    Rmax = Rlimits[1]
    dPMLlo = Rmin - np.min(Rvec)
    dPMLhi = np.max(Rvec) - Rmax

    s = np.zeros_like(Rvec)
    iL = np.argwhere(Rvec < Rmin)
    if (dPMLlo > 0) and not iL:
        sL = sigmaMAX * ((Rmin-Rvec) / dPMLlo)**2
        s[iL] = sL[iL]
    iR = np.argwhere(Rvec > Rmax)
    if (dPMLhi > 0) and not iR:
        sR = sigmaMAX * ((Rvec-Rmax) / dPMLhi) ** 2
        s[iR] = sR[iR]

    return s

def rcomplex(Rvec:np.ndarray, Rlimits:list, sigmaMAX:float):
    """
    Complex r-coordinate component within PML layer.
    """
    Rmin = Rlimits[0]
    Rmax = Rlimits[1]
    dPMLlo = Rmin - np.min(Rvec)
    dPMLhi = np.max(Rvec) - Rmax

    r = np.zeros_like(Rvec)
    iL = np.argwhere(Rvec < Rmin)
    if (dPMLlo > 0) and not iL:
        rL = 1j * sigmaMAX/3 * (Rvec-Rmin)**3 / dPMLlo**2
        r[iL] = rL[iL]
    iR = np.argwhere(Rvec > Rmax)
    if (dPMLhi > 0) and not iR:
        rR = 1j * sigmaMAX/3 * (Rvec-Rmax)**3 / dPMLhi**2
        r[iR] = rR[iR]

    return r

def m2dpmloperR2(er:np.ndarray, Pr:np.ndarray, Pz:np.ndarray, rho:np.ndarray, dr:float, dz:float, keig:float, OPTS=None):
    """
    Matrix operator generator (core engine).
    """

    # Default values for OPTS
    if OPTS is None:
        OPTS['eigmode'] = 'b'
        OPTS['fieldmode'] = 'V'
        OPTS['BC'] = [PEC, PEC, PEC, PEC]

    if not 'BC' in OPTS:
        OPTS['BC'] = [PEC, PEC, PEC, PEC]
    if OPTS['BC'] is None:
        OPTS['BC'] = [PEC, PEC, PEC, PEC]
    BCl = OPTS['BC'][0]
    BCr = OPTS['BC'][1]
    BCd = OPTS['BC'][2]
    BCu = OPTS['BC'][3]
    
    if not 'TWOD' in OPTS:
        OPTS['TWOD'] = 0
    if OPTS['TWOD'] is None:
        OPTS['TWOD'] = 0

    M = round((er.shape[0]+1) / 2)
    N = round((er.shape[1]+1) / 2)

    # Resolve the 'double-density' dielectric matrices into rr, ff, and zz parts
    er_rr, er_ff, er_zz, _      = mresolve(er)
    Pr_rr, Pr_zr, Pr_zz, Pr_rz  = mresolve(Pr)
    Pz_rr, Pz_zr, Pz_zz, Pz_rz  = mresolve(Pz)
    rho_rr, rho_zr, rho_zz, _   = mresolve(rho)

    # Pre-compute some handy values
    dr2 = dr**2
    dz2 = dz**2
    drdz = dr*dz

    # Hrr submatrix diagonals
    A = -1/dr2 * Pr_rr / (rho_rr**2) * int(not OPTS['TWOD'])
    B = rho_zr * Pr_zr / er_ff
    C = rho_rr * er_rr

    # Hrr element 1 (drdr part)
    Hrr_l   = A[:,0:N]  * B[0:M-1,0:N]  * np.vstack((C[M-2:M-1,0:N], C[0:M-2,0:N]))
    Hrr_cl  = -A[:,0:N] * B[0:M-1,0:N]  * C[:,0:N]
    Hrr_r   = A[:,0:N]  * B[1:M,0:N]    * np.vstack((C[1:M-1,0:N], C[0:1,0:N]))
    Hrr_cr  = -A[:,0:N] * B[1:M,0:N]    * C[:,0:N]

    del A, B, C

    # Initialize Hrr boundary conditions
    Hrr_lp  = np.zeros_like(Hrr_l)
    Hrr_rp  = np.zeros_like(Hrr_r)
        
    if BCl == PBC:
        Hrr_lp[0,0:N-1] = Hrr_l[0,0:N-1]
        Hrr_l[0,:] = 0
    elif BCl == PMC or BCl == PMCh:
        Hrr_l[0,:] = 0
        if BCl == PMC:
            Hrr_cl[0,:] = 2 * Hrr_cl[0,:]
    elif BCl == PEC or BCl == PECh:
        Hrr_l[0,:] = 0
        Hrr_cl[0,:] = 0
        if BCl == PECh:
            Hrr_r[0,:] = 2 * Hrr_r[0,:]
            Hrr_cr[0,:] = 2 * Hrr_cr[0,:]
    else:
        raise ValueError(f'Unknown left boundary condition type {BCl}.')
    
    if BCr == PBC:
        Hrr_rp[M-2,0:N-1] = Hrr_r[M-2,0:N-1]
        Hrr_r[M-2,:] = 0
    elif BCr == PMC or BCr == PMCh:
        Hrr_r[M-2,:] = 0
        if BCr == PMC:
            Hrr_cr[M-2,:] = 2 * Hrr_cr[M-2,:]
    elif BCr == PEC or BCr == PECh:
        Hrr_r[M-2,:] = 0
        Hrr_cr[M-2,:] = 0
        if BCr == PECh:
            Hrr_l[M-2,:] = 2 * Hrr_l[M-2,:]
            Hrr_cl[M-2,:] = 2 * Hrr_cl[M-2,:]
    else:
        raise ValueError(f'Unknown right boundary condition type {BCr}.')
    
    # Hrr element 2 (dzdz part)
    Hrr_u = -1/dz2 * Pz_rr[:,0:N] * np.hstack((Pz_rz[:,0:N-1], Pz_rz[:,0:1]))
    Hrr_cu = -Hrr_u
    Hrr_d = -1/dz2 * Pz_rr[:,0:N] * np.hstack((Pz_rz[:,N-2:N-1], Pz_rz[:,0:N-1]))
    Hrr_cd = -Hrr_d

    if BCu == PBC:
        Hrr_u[:,N-1]  = 0
        Hrr_cu[:,N-1] = 0
        Hrr_cd[:,N-1] = 0
        Hrr_d[:,N-1]  = 0
        Hrr_l[:,N-1]  = 0
        Hrr_cl[:,N-1] = 0
        Hrr_cr[:,N-1] = 0
        Hrr_r[:,N-1]  = 0
    elif BCu == PMC or BCu == PMCh:
        Hrr_u[:,N-1]  = 0
        Hrr_cu[:,N-1] = 0
        if BCu == PMC:
            Hrr_d[:,N-1] = 2 * Hrr_d[:,N-1]
            Hrr_cd[:,N-1] = 2 * Hrr_cd[:,N-1]
    elif BCu == PEC or BCu == PECh:
        Hrr_u[:,N-1]  = 0
        Hrr_cu[:,N-1] = 0
        Hrr_cd[:,N-1] = 0
        Hrr_d[:,N-1]  = 0
        Hrr_l[:,N-1]  = 0
        Hrr_cl[:,N-1] = 0
        Hrr_cr[:,N-1] = 0
        Hrr_r[:,N-1]  = 0
        Hrr_u[:,N-2] = 0
        if BCu == PECh:
            Hrr_cd[:,N-2] = 2 * Hrr_cd[:,N-2]
    else:
        raise ValueError(f'Unknown upper boundary condition type {BCu}.')

    if BCd == PBC:
        pass
    elif BCd == PMC or BCd == PMCh:
        Hrr_d[:,0]  = 0
        Hrr_cd[:,0] = 0
        if BCd == PMC:
            Hrr_u[:,0] = 2 * Hrr_u[:,0]
            Hrr_cd[:,0] = 2 * Hrr_cd[:,1]
    elif BCd == PEC or BCd == PECh:
        Hrr_u[:,0]  = 0
        Hrr_cu[:,0] = 0
        Hrr_cd[:,0] = 0
        Hrr_d[:,0]  = 0
        Hrr_l[:,0]  = 0
        Hrr_cl[:,0] = 0
        Hrr_cd[:,0] = 0
        Hrr_r[:,0]  = 0
        if BCd == PECh:
            Hrr_cd[:,1] = 2 * Hrr_cd[:,1]
    else:
        raise ValueError(f'Unknown lower boundary condition type {BCd}.')
    
    # Construct center-field element of 5-point stencil block
    Hrr_c = Hrr_cl + Hrr_cr + Hrr_cu + Hrr_cd

    del Hrr_cl, Hrr_cr, Hrr_cu, Hrr_cd

    # Contruct complete Hrr operator block
    Hrr_data = np.vstack((
        rowize(Hrr_u)*(BCu==PBC), 
        rowize(Hrr_d), 
        rowize(Hrr_rp), 
        rowize(Hrr_l), 
        rowize(Hrr_c), 
        rowize(Hrr_r), 
        rowize(Hrr_lp), 
        rowize(Hrr_u), 
        rowize(Hrr_d)*(BCd==PBC)))
    Hrr_diags = np.array([-(M-1)+(M-1)*(N-1), M-1, M-2, 1, 0, -1, -(M-2), -(M-1), (M-1)-(M-1)*(N-1)])
    Hrr = spdiags2(
        data=Hrr_data, 
        diags=Hrr_diags,
        m=(M-1)*N, 
        n=(M-1)*N)
    Hrr = Hrr.T

    del Hrr_l, Hrr_r, Hrr_u, Hrr_d, Hrr_lp, Hrr_rp, Hrr_data, Hrr_diags

    # Hzz submatrix diagonals
    A1 = Pr_zz / dr2 / rho_zz
    B = -Pz_zz/dz2 * int(not OPTS['TWOD'])
    C = Pz_zr / er_ff

    Hzz_l = -A1[0:M,:] * np.vstack((Pr_rz[M-2:M-1,:], Pr_rz[0:M-1,:])) * np.vstack((rho_rr[M-2:M-1,0:N-1], rho_rr[0:M-1,0:N-1]))
    Hzz_cl = -Hzz_l
    Hzz_r = -A1[0:M,:] * np.vstack((Pr_rz[0:M-1,:], Pr_rz[0:1,:])) * np.vstack((rho_rr[0:M-1,0:N-1], rho_rr[0:1,0:N-1]))
    Hzz_cr = -Hzz_r

    del A

    Hzz_u = B[0:M,:] * C[0:M,1:N] * np.hstack((er_zz[0:M,1:N-1], er_zz[0:M,0:1]))
    Hzz_cu = -B[0:M,:] * C[0:M,1:N] * er_zz[0:M,:]
    Hzz_d = B[0:M,:] * C[0:M,0:N-1] * np.hstack((er_zz[0:M,N-2:N-1], er_zz[0:M,0:N-2]))
    Hzz_cd = -B[0:M,:] * C[0:M,0:N-1] * er_zz[0:M,:]

    del B, C

    # Initialize Hzz boundary conditions
    Hzz_lp = np.zeros_like(Hzz_l)
    Hzz_rp = np.zeros_like(Hzz_r)

    if BCl == PBC:
        Hzz_lp[0,:] = Hzz_l[0,:]
        Hzz_l[0,:] = 0
    elif BCl == PMC or BCl == PMCh:
        Hzz_l[0,:] = 0
        Hzz_cl[0,:] = 0
        if BCl == PMC:
            Hzz_r[0,:] = 2 * Hzz_r[0,:]
            Hzz_cr[0,:] = 2 * Hzz_cr[0,:]
    elif BCl == PEC or BCl == PECh:
        Hzz_l[0,:]  = 0
        Hzz_cl[0,:] = 0
        Hzz_cr[0,:] = 0
        Hzz_r[0,:]  = 0
        Hzz_u[0,:]  = 0
        Hzz_cu[0,:] = 0
        Hzz_cd[0,:] = 0
        Hzz_d[0,:]  = 0
        Hzz_l[1,:]  = 0
        if BCl == PECh:
            Hzz_cl[1,:] = 2 * Hzz_cl[1,:]
    else:
        raise ValueError(f'Unknown left boundary condition type {BCl}.')
    
    if BCr == PBC:
        Hzz_l[M-1,:]  = 0
        Hzz_cl[M-1,:] = 0
        Hzz_cr[M-1,:] = 0
        Hzz_r[M-1,:]  = 0
        Hzz_u[M-1,:]  = 0
        Hzz_cu[M-1,:] = 0
        Hzz_cd[M-1,:] = 0
        Hzz_d[M-1,:]  = 0
        Hzz_rp[M-2,:] = Hzz_r[M-2,:]
        Hzz_r[M-2,:] = 0
    elif BCr == PMC or BCr == PMCh:
        Hzz_r[M-1,:] = 0
        Hzz_cr[M-1,:] = 0
        if BCr == PMC:
            Hzz_l[M-1,:] = 2 * Hzz_l[M-1,:]
            Hzz_cl[M-1,:] = 2 * Hzz_cl[M-1,:]
    elif BCr == PEC or BCr == PECh:
        Hzz_l[M-1,:]  = 0
        Hzz_cl[M-1,:] = 0
        Hzz_cr[M-1,:] = 0
        Hzz_r[M-1,:]  = 0
        Hzz_u[M-1,:]  = 0
        Hzz_cr[M-1,:] = 0
        Hzz_cd[M-1,:] = 0
        Hzz_d[M-1,:]  = 0
        Hzz_r[M-2,:] = 0
        if BCr == PECh:
            Hzz_cr[M-2,:] = 2 * Hzz_cr[M-2,:]
    else:
        raise ValueError(f'Unknown right boundary condition type {BCr}.')
    
    if BCu == PBC:
        pass
    elif BCu == PMC or BCu == PMCh:
        Hzz_u[:,N-2] = 0
        if BCu == PMC:
            Hzz_cu[:,N-2] = 2 * Hzz_cu[:,N-2]
    elif BCu == PEC or BCu == PECh:
        Hzz_u[:,N-2] = 0
        Hzz_cu[:,N-2] = 0
        if BCu == PECh:
            Hzz_d[:,N-2] = 2 * Hzz_d[:,N-2]
            Hzz_cd[:,N-2] = 2 * Hzz_cd[:,N-2]
    else:
        raise ValueError(f'Unknown upper boundary condition type {BCu}.')
    
    if BCd == PBC:
        pass
    elif BCd == PMC or BCd == PMCh:
        Hzz_d[:,1] = 0
        if BCd == PMC:
            Hzz_cd[:,1] = 2 * Hzz_cd[:,1]
    elif BCd == PEC or BCd == PECh:
        Hzz_d[:,1] = 0
        Hzz_cd[:,1] = 0
        if BCd == PECh:
            Hzz_u[:,1] = 2 * Hzz_u[:,1]
            Hzz_cu[:,1] = 2 * Hzz_cu[:,1]
    else:
        raise ValueError(f'Unknown lower boundary condition type {BCd}.')
    
    # Construct center-field element of 5-point stencil block
    Hzz_c = Hzz_cl + Hzz_cr + Hzz_cu + Hzz_cd

    del Hzz_cl, Hzz_cr, Hzz_cu, Hzz_cd

    # Contruct complete Hrr operator block
    Hzz_data = np.vstack((
        rowize(Hzz_u), 
        rowize(Hzz_d), 
        rowize(Hzz_rp), 
        rowize(Hzz_l),  
        rowize(Hzz_c), 
        rowize(Hzz_r), 
        rowize(Hzz_lp), 
        rowize(Hzz_u), 
        rowize(Hzz_d)))
    Hzz_diags = np.array([-M+M*(N-1), M, M-2, 1, 0, -1, -(M-2), -M, M-M*(N-1)])
    Hzz = spdiags2(
        data=Hzz_data, 
        diags=Hzz_diags, 
        m=M*(N-1), 
        n=M*(N-1))
    Hzz = Hzz.T

    del Hzz_l, Hzz_r, Hzz_u, Hzz_d, Hzz_lp, Hzz_rp, Hzz_data, Hzz_diags
    
    # Off-diagonal submatrices only for full vector solutions
    if OPTS['fieldmode'] == 'V':

        # Hrz submatrix diagonals
        A1 = Pz_rr[:,0:N] / drdz
        A2 = A1 * np.hstack((Pr_rz[:,0:N-1], Pr_rz[:,0:1]))
        B = Pr_rr[:,0:N] / drdz / rho_rr[:,0:N]**2
        C = rho_zr[:,0:N]**2 * Pz_zr[:,0:N] / er_ff[:,0:N]

        Hrz_lu = -A2 + B * C[0:M-1,0:N] * np.hstack((er_zz[0:M-1,0:N-1], er_zz[0:M-1,0:1]))
        Hrz_ru = A2 - B * C[1:M,0:N] * np.hstack((er_zz[1:M,0:N-1], er_zz[1:M,0:1]))

        A2 = A1 * np.hstack((Pr_rz[:,N-2:N-1], Pr_rz[:,0:N-1]))

        Hrz_ld = A2 - B * C[0:M-1,0:N] * np.hstack((er_zz[0:M-1,N-2:N-1], er_zz[0:M-1,0:N-1]))
        Hrz_rd = -A2 + B * C[1:M,0:N] * np.hstack((er_zz[1:M,N-2:N-1], er_zz[1:M,0:N-1]))

        del A1, A2, B, C

        # Initialize Hrz boundary conditions
        Hrz_rdp = np.zeros_like(Hrz_rd)
        Hrz_rup = np.zeros_like(Hrz_ru)

        if BCl == PBC or BCl == PMC or BCl == PMCh:
            pass
        elif BCl == PEC or BCl == PECh:
            Hrz_lu[0,:] = 0
            Hrz_ld[0,:] = 0
            if BCl == PECh:
                Hrz_ru[0,:] = 2 * Hrz_ru[0,:]
                Hrz_rd[0,:] = 2 * Hrz_rd[0,:]
        else:
            raise ValueError(f'Unknown left boundary condition type {BCl}.')
        
        if BCr == PBC:
            Hrz_rdp[M-2,:]  = Hrz_rd[M-2,:]
            Hrz_rd[M-2,:]   = 0
            Hrz_rup[M-2,:]  = Hrz_ru[M-2,:]
            Hrz_ru[M-2,:]   = 0
        elif BCr == PMC or BCr == PMCh:
            pass
        elif BCr == PEC or BCr == PECh:
            Hrz_ru[M-2,:] = 0
            Hrz_rd[M-2,:] = 0
            if BCr == PECh:
                Hrz_lu[M-2,:] = 2 * Hrz_lu[M-2,:]
                Hrz_ld[M-2,:] = 2 * Hrz_ld[M-2,:]
        else:
            raise ValueError(f'Unknown right boundary condition type {BCr}.')
        
        if BCu == PMC or BCu == PMCh:
            Hrz_lu[:,N-1] = 0
            Hrz_ru[:,N-1] = 0
            if BCu == PMC:
                Hrz_ld[:,N-1] = 2 * Hrz_ld[:,N-1]
                Hrz_rd[:,N-1] = 2 * Hrz_rd[:,N-1]
        elif BCu == PBC or BCu == PEC or BCu == PECh:
            Hrz_lu[:,N-1] = 0
            Hrz_ru[:,N-1] = 0
            Hrz_ld[:,N-1] = 0
            Hrz_rd[:,N-1] = 0
        else:
            raise ValueError(f'Unknown upper boundary condition type {BCu}.')
        
        if BCd == PMC:
            pass
        elif BCd == PMC or BCd == PMCh:
            Hrz_ld[:,0] = 0
            Hrz_rd[:,0] = 0
            if BCd == PMC:
                Hrz_lu[:,0] = 2 * Hrz_lu[:,0]
                Hrz_ru[:,0] = 2 * Hrz_ru[:,0]
        elif BCd == PEC or BCd == PECh:
            Hrz_lu[:,0] = 0
            Hrz_ru[:,0] = 0
            Hrz_ld[:,0] = 0
            Hrz_rd[:,0] = 0
        else:
            raise ValueError(f'Unknown lower boundary condition type {BCd}.')
        
        # Fill left/right with zeros to match the input Ez field size before diagonalizing block
        Hrz_lu  = np.vstack(Hrz_lu, np.zeros((1,N)))
        Hrz_ru  = np.vstack(Hrz_ru, np.zeros((1,N)))
        Hrz_rup = np.vstack(Hrz_rup, np.zeros((1,N)))
        Hrz_ld  = np.vstack(Hrz_ld, np.zeros((1,N)))
        Hrz_rd  = np.vstack(Hrz_rd, np.zeros((1,N)))
        Hrz_rdp = np.vstack(Hrz_rdp, np.zeros((1,N)))

        Hrz = spdiags2(
            data=np.hstack((Hrz_rdp*(BCr==PBC), Hrz_ld, Hrz_rd, Hrz_rup*(BCr==PBC), Hrz_lu, Hrz_ru, Hrz_ld*(BCd==PBC), Hrz_rd*(BCd==PBC))), 
            diags=[M+M-2, M, M-1, M-2, 0, -1, -(N-2)*M, -((N-2)*M+1)], 
            m=M*(N-1), 
            n=M*N)
        Hrz = Hrz.T
        Hrz[M-2,(N-2)*M] = Hrz_rd[M-2,0] * (BCr==PBC) * (BCd==PBC)

        del Hrz_ld, Hrz_lu, Hrz_rd, Hrz_ru, Hrz_rdp, Hrz_rup

        nidx = np.setdiff1d(np.arange(M*N), M*np.arange(N))
        Hrz = Hrz[nidx,:]

        A1 = Pr_zz[0:M,:] / drdz / rho_zz[0:M,:]
        A2 = A1 * np.vstack((Pz_rz[0:M-1,:], Pz_rz[0:1,:]))
        B1 = Pz_zz[0:M,:] / drdz / rho_zz[0:M,:]
        B2 = B1 * Pr_zr[0:M,0:N-1] / er_ff[0:M,0:N-1]
        C = rho_rr * er_rr

        Hzr_rd = -A2 + B2 * np.vstack((C[0:M-1,0:N-1], C[0:1,0:N-1]))

        A1 = A1 * np.vstack((Pz_rz[M-2:M-1,:], Pz_rz[0:M-1,:])) * np.vstack((rho_rr[M-2:M-1,0:N-1], rho_rr[0:M-1,0:N-1]))

        Hzr_ld = A1 - B2 * np.vstack((C[M-2:M-1,0:N-1], C[0:M-1,0:N-1]))
        B1 = B1 * Pr_zr[0:M,1:N] / er_ff[0:M,1:N]
        Hzr_ru = A2 - B1 * np.vstack((C[0:M-1,1:N], C[0:1,1:N]))
        Hzr_lu = -A1 + B1 * np.vstack((C[M-2:M-1,1:N], C[0:M-1,1:N]))

        del A1, A2, B1, B2, C

        # Set up Hzr boundary conditions
        Hzr_ldp = np.zeros_like(Hzr_ld)
        Hzr_lup = np.zeros_like(Hzr_lu)

        if BCd == PBC or BCd == PMC or BCd == PMCh:
            pass
        elif BCd == PEC or BCd == PECh:
            Hzr_ld[:,0] = 0
            Hzr_rd[:,0] = 0
            if BCd == PECh:
                Hzr_ld[:,0] = 2 * Hzr_lu[:,0]
                Hzr_ru[:,0] = 2 * Hzr_ru[:,0]
        else:
            raise ValueError(f'Unknown lower boundary condition type {BCd}.')
        
        if BCu == PBC:
            pass
        elif BCu == PMC or BCu == PMCh:
            pass
        elif BCu == PEC or BCu == PECh:
            Hzr_lu[:,N-2] = 0
            Hzr_ru[:,N-2] = 0
            if BCu == PECh:
                Hzr_ld[:,N-2] = 2 * Hzr_ld[:,N-2]
                Hzr_rd[:,N-2] = 2 * Hzr_rd[:,N-2]
        else:
            raise ValueError(f'Unknown upper boundary condition type {BCu}.')
        
        if BCr == PMC or BCr == PMCh:
            Hzr_ru[M-1,:] = 0
            Hzr_rd[M-1,:] = 0
            if BCr == PMC:
                Hzr_lu[M-1,:] = 2 * Hzr_lu[M-1,:]
                Hzr_ld[M-1,:] = 2 * Hzr_ld[M-1,:]
        elif BCr == PBC or BCr == PEC or BCr == PECh:
            Hzr_lu[M-1,:] = 0
            Hzr_ld[M-1,:] = 0
            Hzr_ru[M-1,:] = 0
            Hzr_ru[M-1,:] = 0
        else:
            raise ValueError(f'Unknown right boundary condition type {BCr}.')
        
        if BCl == PBC:
            Hzr_ldp[0,:] = Hzr_ld[0,:]
            Hzr_ld[0,:] = 0
            Hzr_lup[0,:] = Hzr_lu[0,:]
            Hzr_lu[0,:] = 0
        elif BCl == PMC or BCl == PMCh:
            Hzr_lu[0,:] = 0
            Hzr_ld[0,:] = 0
            if BCl == PMC:
                Hzr_ru[0,:] = 2 * Hzr_ru[0,:]
                Hzr_rd[0,:] = 2 * Hzr_rd[0,:]
        elif BCl == PEC or BCl == PECh:
            Hzr_lu[0,:] = 0
            Hzr_ld[0,:] = 0
            Hzr_ru[0,:] = 0
            Hzr_rd[0,:] = 0
        else:
            raise ValueError(f'Unknown left boundary condition type {BCl}.')
        
        Hzr_data = np.vstack((
            rowize(Hzr_lu*(BCu==PBC)), 
            rowize(Hzr_ru*(BCu==PBC)), 
            rowize(Hzr_lu), 
            rowize(Hzr_rd), 
            rowize(Hzr_ldp*(BCr==PBC)), 
            rowize(Hzr_lu), 
            rowize(Hzr_ru), 
            rowize(Hzr_lup*(BCr==PBC))))
        Hzr_diags = np.array([(N-2)*M+1, (N-2)*M, 1, 0, -(M-2), -(M-1), -M, -2*(M-1)])
        Hzr = spdiags2(
            data=Hzr_data, 
            diags=Hzr_diags, 
            m=M*N, 
            n=M*(N-1))
        Hzr = Hzr.T
        Hzr[(N-2)*M,M-2] = Hzr_lup[0,N-2] * (BCr == PBC) * (BCu == PBC)

        del Hzr_ld, Hzr_lu, Hzr_rd, Hzr_ru, Hzr_ldp, Hzr_lup

        nidx = np.setdiff1d(np.arange(M*N), M*np.arange(N))
        Hzr = Hzr[:,nidx]
    else:
        Hzr = lil_matrix((Hzz.shape[0], Hzz.shape[1]))
        Hzr = Hrz.T

    if OPTS['eigmode'] == 'w':
        A = rho_rr[:,0:N]**2
        Hrr = Hrr + spdiags2(data=1/A, diags=0, m=Hrr.shape[0], n=Hrr.shape[1]) * keig

        A = rho_zz[0:M,:]**2
        Hzz = Hzz + spdiags2(data=1/A, diags=0, m=Hzz.shape[0], n=Hzz.shape[1]) * keig

        A = er_rr[:,0:N]
        A = spdiags2(data=1/A, diags=0, m=Hrr.shape[0], n=Hrr.shape[1])
        Hrr = A * Hrr
        Hrz = A * Hrz

        A = er_zz[0:M,:]
        A = spdiags2(data=1/A, diags=0, m=Hzz.shape[0], n=Hzz.shape[1])
        Hzz = A * Hzz
        Hzr = A * Hzr
    else:
        A = rho_rr[:,0:N]**2
        B = er_rr[:,0:N]
        NH = Hrr.shape
        Hrr = spdiags2(data=rowize(A), diags=0, m=NH[0], n=NH[1]) * (-Hrr + spdiags2(data=rowize(B), diags=0, m=NH[0], n=NH[1])*keig)
        Hrz = -spdiags2(data=rowize(A), diags=0, m=NH[0], n=NH[1]) * Hrz

        del A, B, NH

        A = rho_zz[0:M,:]**2
        B = er_zz[0:M,:]
        NH = Hzz.shape
        Hzz = spdiags2(data=rowize(A), diags=0, m=NH[0], n=NH[1]) * (-Hzz + spdiags2(data=rowize(B), diags=0, m=NH[0], n=NH[1])*keig)
        Hzr = -spdiags2(data=rowize(A), diags=0, m=NH[0], n=NH[1]) * Hzr

        del A, B, NH

    ix = []

    if BCd == PEC or BCd == PECh:
        ix = np.hstack((ix, np.arange(M-1)))
    if BCu == PEC or BCu == PECh or BCu == PBC:
        ix = np.hstack((ix, (M-1)*(N-1) + np.arange(M-1)))
    
    ix = np.setdiff1d(np.arange((M-1)*N), ix)

    Hrr = Hrr[ix,ix]
    Hrz = Hrz[ix,:]
    Hzr = Hzr[:,ix]

    ix = []

    if BCl == PEC or BCl == PECh:
        ix = np.hstack((ix, ))
    if BCr == PEC or BCr == PECh or BCr == PBC:
        ix = np.hstack((ix, ))

    ix = np.setdiff1d(np.arange(M*(N-1)), ix)
    Hzz = Hzz[ix,ix]
    Hrz = Hrz[:,ix]
    Hzr = Hzr[ix,:]

    H = np.vstack((np.hstack((Hrr, Hrz)), np.hstack((Hzr, Hzz))))

    del Hrr, Hrz, Hzr, Hzz

    return H

def mresolve(A:np.ndarray):
    """
    Splits large matrix into 3-4 interleaved smaller matrices.
    """
    M = round((A.shape[0]+1) / 2)
    N = round((A.shape[1]+1) / 2)
    P = A.shape[1]
    # if (P == 3):
    #     p = [1, 2, 3]
    # else:
    #     p = [1, 1, 1]

    A_rr = A[1:-1:2, 0:None:2]
    A_zr = A[0:None:2, 0:None:2]
    A_zz = A[0:None:2, 1:-1:2]
    A_rz = A[1:-1:2, 1:-1:2]

    return A_rr, A_zr, A_zz, A_rz

def rowize(A:np.ndarray):
    """
    Convert 2D matrix into row vector
    """
    return np.reshape(A, (1, np.size(A)))

def columnize(A:np.ndarray):
    """
    Convert 2D matrix into column vector.
    """
    return np.reshape(A, (np.size(A), 1))

def spdiags2(data:np.ndarray, diags:np.ndarray, m:int, n:int):
    """
    Modified spdiags function.

    Parameters
    ----------
        data : [M-by-N] ndarray
        diags : [1-by-N] ndarray
        m : int
        n : int
    """
    if m < n:
        res = spdiags(data=data, diags=diags, m=m, n=n).tolil()
        res = res[0:m,:]
        # res = res.todia()
    else:
        res = spdiags(data=data, diags=diags, m=m, n=n).tolil()
    return res

def diffxy(fieldmtx:np.ndarray, fieldtype:str, BClist:list, ddir:int):
    BCl = BClist[0]
    BCr = BClist[1]
    BCd = BClist[2]
    BCu = BClist[3]

    

    dfieldovdn = np.diff(fieldmtx, 1, 1+ddir)

    return dfieldovdn