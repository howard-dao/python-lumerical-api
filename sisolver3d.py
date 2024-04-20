import numpy as np

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
        mu_guess = OPTS.mu_guess
    else:
        mu_guess = max(nlayers) * k0

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

    xlayers[-1] = round(np.sum(xlayers)/dx)*dx - np.sum(xlayers[0:-2])
    ylayers[-1] = round(np.sum(ylayers)/dy)*dy - np.sum(ylayers[0:-2])

    # Interface (?)
    xint = np.insert(np.cumsum(xlayers), 0, 0)
    yint = np.insert(np.cumsum(ylayers), 0, 0)

    # Grid coordinates
    x = np.arange(0, xint[-1], dx/2)
    y = np.arange(0, xint[-1], dy/2)

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

            npsqz[ixin0:ixin1+1, iyin0:iyin1+1] = npsqz[ixin0:ixin1+1, iyin0:iyin1+1] + (xpmax-xpmin) @ np.transpose(ypmax-ypmin) / (dx*dy) * nlayers[nn,mm]**2

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

            npsqy[ixin0:ixin1+1, iyin0:iyin1+1] = 1 / (1/npsqy[ixin0:ixin1+1, iyin0:iyin1+1] + 1 / ((xpmax-xpmin) @ np.transpose(1/(ypmax-ypmin)) / (dx/dy) * nlayers[nn,mm]**2))

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

            npsqx[ixin0:ixin1+1, iyin0:iyin1+1] = 1 / (1/npsqx[ixin0:ixin1+1, iyin0:iyin1+1] + 1 / (1 / (xpmax-xpmin) @ np.transpose(ypmax-ypmin) / (dy/dx) * nlayers[nn,mm]**2))

        npsqx[np.isinf(npsqx)] = 0
        npsqxtot = npsqxtot + npsqx
        npsqx = np.inf * np.ones((len(xp), len(yp)))

    N['n'][0:None:2, 0:None:2] = np.sqrt(npsqz)
    N['n'][0:None:2, 1:-1:2] = np.sqrt(npsqytot)
    N['n'][1:-1:2, 0:None:2] = np.sqrt(npsqxtot)

    N['n'][1:-1:2, 1:-1:2] = np.sqrt((N['n'][0:-2:2, 1:-1:2]**2 + N['n'][2:None:2, 1:-1:2]**2 + N['n'][1:-1:2, 0:-2:2]**2 + N['n'][1:-1:2, 2:None:2]**2) / 4)

    if np.argwhere(np.isinf(N['n'])):
        raise ValueError('Si solver error: Generated index distribution has non-fininte elements.')

    return N