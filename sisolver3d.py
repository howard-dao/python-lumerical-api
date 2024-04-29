import numpy as np
from scipy.constants import speed_of_light as c
from scipy.constants import mu_0 as mu0

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
    N['x'] = x
    N['y'] = y
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
    
    # # For bend modes
    # if OPTS['radius'] in OPTS:
    #     R = OPTS['radius']
    #     N['x'] = N['x'] + R
    #     OPTS['coordmode'] = 'R'
    
    # # Call mode solver engine with prepared parameters
    # beta, F, V = eval(OPTS['enginever'], N, k0, mu_guess, OPTS, NMODES_CALC, OPTS['PMLwidth'], OPTS['PMLsigma'])
    
    # if OPTS['coordmode'] == 'R':
    #     alpha = np.imag(beta) / R
    #     F['dB90'] = 20 * alpha * (np.pi*R/2) * np.log10(np.exp(1))
    # else:
    #     alpha = np.imag(beta)

    # F['dBcm'] = 0.2 / np.log(10) * alpha
    # F['beta'] = beta
    # F['LossQ'] = np.real(beta) / np.imag(beta) / 2

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
    PEC     = 0
    PMCh    = 1
    PBC     = 2
    PMC     = 3
    PECh    = 4

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

    R = S.x
    Z = S.y
    nn = S.n

    L = nu_index

    sigmaRMAX = sigmaPML[0]
    sigmaZMAX = sigmaPML[1]

    NR = (len(R) + 1) / 2
    NZ = (len(Z) + 1) / 2

    w0 = c * nu_index

    Pr, Pz, Rmtx = m2dpmlmatx(w0, nn**2, R, Z, [sigmaRMAX, sigmaZMAX], dPML)

    dR = R[2] - R[0]
    dZ = Z[2] - Z[0]

    if OPTS['coordmode'] == 'C':
        Rmtx = np.ones(np.shape(Rmtx))

    H = eval(OPTS['operver'], nn**2, Pr, Pz, Rmtx, dR, dZ, L**2, OPTS)
    H = H / mu_guess[0]**2

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
        H = np.transpose(H)
    
    asgn = (-1)**OPTS['adjoint']

    V,D = eval(OPTS['eigsfcn'], H, nmodes, OPTS['sigma'], OPTS)

    mu = np.sqrt(np.diag(D)) * mu_guess[0]

    if OPTS['eigmode'] == 'w':
        X = np.sortrows([-np.imag(mu)*asgn, np.real(mu), np.transpose(V)])

        mu = -X[:,1] + 1j*X[:,0]*asgn

    else:
        X = np.sortrows(np.imag(mu)*asgn, -np.real(mu), np.transpose(V))

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



def m2dpmlmatx():
    return

def diffxy(fieldmtx:np.ndarray, fieldtype:str, BClist:list, ddir:int):
    PEC = 0
    PMCh = 1
    PBC = 2
    PMC = 3
    PECh = 4

    BCl = BClist[0]
    BCr = BClist[1]
    BCd = BClist[2]
    BCu = BClist[3]

    

    dfieldovdn = np.diff(fieldmtx, 1, 1+ddir)

    return dfieldovdn