#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  opt.py
#
#  Copyright 2017-2018 Alexei Andreanov <alexei@pcs.ibs.re.kr>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


#   Generic sequential optimisation based flatband engineering

import math as _mt
import cmath as _cmt

import numpy as _np
import cvxopt as _co
import scipy.linalg as _la

import fake.linalg as _fla

#   Defnitions
#   Internal


#   Public
slp_type_all = ('lndet', 'det')                                         # available/implemented sequential LP solvers

#   Generic SLP routines
def slp_init(config, T_all, couplings, q_num):
    """
    SLP initialisation

    The layout of the constraint matrix
        Rows                                    Type of constraint
        0..4q_num-1                                 wavevetor related
        4q_num                                      the positivity of c
        4q_num + 1..4q_num + 1 + 2*v_num            the bounds on the
                                                        variables

    Parameters
    ----------
    slp_t : str
    T_all : array

    Returns
    -------
    aux : dict
        various auxiliary information
    """
    slp_t = config['slp_type']                                          # type of sequential LP optimisation
    no_target = not ('E' in config)                                     # target FB energy not specified

    nu = T_all.shape[1]                                                 # number of bands

#   ?
    aux = {'T_all': T_all, 'Hq': _np.zeros((nu, nu), dtype=_np.complex128)}
    aux['Cq'] = _np.zeros((nu, nu), dtype=_np.complex128)               # cofactor matrix
    aux['I'] = _np.eye(nu, dtype=_np.complex128)                        # identity matrix

    aux['slp_type'] = slp_t                                             #
    aux['eps'] = 1e-8                                                   # zero threshold
    aux['no_target'] = no_target                                        #

#   ?
    g_num = len(couplings)                                              # number of couplings
    v_num = 1 + g_num                                                   # number of LP variables: c + couplings
    if no_target:                                                       # target FB energy not specified
        v_num += 1                                                      # FB energy is also a variable

#   Cost function
    cf = _co.matrix(_np.zeros(v_num))
    cf[0] = 1.0                                                         # the constraint to be minimised

#   Number of constraints
    if slp_t == 'lndet':                                                # lndet LP
        c_num = q_num                                                   # number of constraints

#    dE = 0.0                                                            #
        aux['dE_min'] = 0.0                                             #
        aux['dE_all'] = _np.zeros(q_num)                                #
#    r = 0.0

        aux['G0'] = _np.zeros((q_num, v_num - 1))                       # regular part of the constraint matrix; "c" column dropped
        aux['G0s'] = _np.zeros_like(G0)                                 # singular part of the constraint matrix
    elif slp_t == 'det':                                                # det LP
        c_num = 4*q_num + 1                                             # number of constraints: wavevectors + (c>=0)

        if 'dc_bound' in config:                                        # couplings shifts bounds
#            aux['dc_bound'] = config['dc_bound']

            c_num += 2*g_num                                            # |g| <= the bound

        if 'dE_bound' in config:                                        # FB energy shift bound
#            aux['dE_bound'] = config['dE_bound']

            c_num += 2                                                  # |E| <= the bound

#   ?
    aux['v_num'] = v_num                                                # number of LP variables
    aux['c_num'] = c_num                                                # number of LP constraints

#   Allocation
    G = _co.matrix(_np.zeros((c_num, v_num)))                           # RHS, the constraint matrix
    h = _co.matrix(_np.zeros(c_num))                                    # LHS

#   ?
    if slp_t == 'lndet':                                                # lndet LP
        G[:, 0] = -1.0
    elif slp_t == 'det':                                                # det LP
        for i in range(4*q_num):                                        # loop over the wavevectors
            G[i, 0] = -1.0
        G[4*q_num, 0] = -1.0                                            # c >= 0 <=> -c <= 0

        if 'dc_bound' in config:                                        #
            dc = config['dc_bound']                                     # the bound

            for i in range(g_num):                                      # loop over the couplings
                G[4*q_num + 1 + 2*i, 1 + i] = -1.0
                G[4*q_num + 1 + 2*i + 1, 1 + i] = 1.0

                h[4*q_num + 1 + 2*i] = dc
                h[4*q_num + 1 + 2*i + 1] = dc
        if 'dE_bound' in config:                                        #
            dE = config['dE_bound']                                     # the bound

            if 'dc_bound' in config:                                    #
                i0 = 4*q_num + 1 + 2*g_num
            else:
                i0 = 4*q_num + 1

            G[i0, -1] = -1.0
            G[i0 + 1, -1] = 1.0

            h[i0] = dE
            h[i0 + 1] = dE

    return (cf, G, h, aux)

def slp_update(sol, E_cur, couplings, couplings_old, aux):
    """
    SLP intermediate update

    Parameters
    ----------
    couplings : array

    Returns
    -------
    err : float
        convergence error
    """
    slp_t = aux['slp_type']                                             #

    if slp_t == 'ln_det':                                               #
        g_num = len(couplings)
        dE_min = aux['dE_min']

        couplings = dE_min*sol[1:1+g_num]                               # compute the new couplings
        if no_target:                                                   # if flatband energy is not specified
            E_cur += dE_min*sol[2 + g_num]                              # update the current would be FB energy

        err = _np.max(_np.abs(couplings - couplings_old))               # convergence error
    elif slp_t == 'det':                                                #
        err = sol[0]                                                    # convergence error

    return (E_cur, err)

def slp_finalise():
    """
    SLP finalisation
    """
    pass

#   Log determinant based LP
def constraints_lndet_make(G, h, H_func, q_all, couplings, E_cur, aux):
    """
    Compute the constraint matrix for the ln based sequential LP
    optimisation

    Parameters
    ----------
    G : array
    h : array
    H_func : func
    q_all : array
        wavevectors
    couplings : array
        current couplings/hoppings
    E_cur : float
        current value of the FB energy
    aux : dict
        all the auxiliary data
    """
#   ?
    T_all = aux['T_all']                                                #
    Hq = aux['Hq']                                                      # the current Bloch Hamiltonian
    Cq = aux['Cq']                                                      #
    dE_all = aux['dE_all']                                              #

#   ?
    dE = 0.
    adE_min = 0.

#   ?
    G0 = _np.zeros((q_num, v_num - 1))                                  # regular part of the constraint matrix; "c" column dropped
    G0s = _np.zeros_like(G0)                                            # singular part of the constraint matrix
    G = _co.matrix(_np.zeros((q_num, v_num)))                           # RHS, the constraint matrix
    G[:, 0] = -1.0                                                      # set the column of the "c" variable
    h = _co.matrix(_np.zeros(q_num))                                    # LHS

#   Compute the constraints matrix
    i = 0
    for q in q_all:                                                     # loop over all the wavevectors
        Hq[:, :] = H_func(q, couplings) - E_cur*I                       # the Hamiltonian matrix at wavevector q = q_all[i]

        (E, psi) = _la.eigh(Hq)                                         # the spectrum at wavevector q = q_all[i]
        ix = _np.argmin(_np.abs(E))                                     # index of the would be flatband/zero mode
        dE = E[ix]
        dE_all[i] = dE
        if abs(dE) < dE_min:
            adE_min = dE
        u0 = psi[:, ix]                                                 # the eigenmode of the would be zero mode of the Hamiltonian matrix

        if abs(dE) > eig_eps:                                           # no zero modes in the spectrum
            Rq = _la.inv(Hq)                                            #

            for j in range(c_num):
                G0[i, j] = _np.trace(Rq@T_all[j])

            if no_target:                                               # target FB energy not specified
                G0[i, c_num] = -_np.trace(Rq)
        else:                                                           # zero mode present
            Rq = _la.pinv(Hq, cond=eig_eps)                             # CHECK
#                P = _np.outer(psi[:, ix], psi[:, ix])

            for j in range(c_num):
                G0[i, j] = _np.trace(Rq@T_all[j])
                G0s[i, j] = u0@T_all[j]@u0                              # singular contribution

            if no_target:
                G0[i, c_num] = -_np.trace(Rq)
                G0s[i, c_num] = -1.0

        (sign, alndet) = _np.linalg.slogdet(Hq)
        h[i] = -alndet

        i += 1

    aux['dE_min'] = dE_min                                              #

#   Finalise the constraints matrix
    for i in range(q_num):
#        r = adE_min/dE_all[i]

#        G0[i, :] *= dE_min
#        G0s[i, :] *= r
#            G0[i, :c_num] *= dE_min
#            G0s[i, :c_num] *= r
#            if no_target:
#                G0[i, c_num] *= dE_min
#                G0s[i, c_num] *= r

        G[i, 1:] = G0[i, :] + G0s[i, :]

    h[:] /= adE_min

#   Determinant based LP
def constraints_det_make(G, h, H_func, q_all, couplings, E_cur, aux):
    """
    """
#   ?
    T_all = aux['T_all']                                                #
    Hq = aux['Hq']                                                      #
    Cq = aux['Cq']                                                      #
    I = aux['I']                                                        # identity matrix

    eps = aux['eps']                                                    # zero threshold
    no_target = aux['no_target']                                        #
    g_num = len(couplings)                                              # number of couplings

    tc = 0.0j
    tct = 0.0j
    cq = 0.0

#   ?
    i = 0
    j = 0
    for q in q_all:                                                     #
        Hq[:, :] = H_func(q, couplings) - E_cur*I                       #
        Cq[:, :] = _fla.cofactor(Hq)                                    # cofactor matrix of Hq

        det_Hq = _la.det(Hq)                                            # real

        assert _np.abs(det_Hq.imag) < eps, 'Complex determinant of H_q detected!'

        j = 4*i
        cq = 2*_mt.cos(q)

#   LHS - constraint matrix
        for k in range(g_num):                                          # loop over couplings
            tct = 2*cq*_np.trace(Cq@T_all[k])                           # complex

            G[j, 1 + k] = -tct.real
            G[j + 1, 1 + k] = -tct.imag

            G[j + 2, 1 + k] = tct.real
            G[j + 3, 1 + k] = tct.imag

        if not no_target:                                               #
            tc = _np.trace(Cq)                                          # complex

            G[j, 1 + g_num] = tc.real
            G[j + 1, 1 + g_num] = tc.imag

            G[j + 2, 1 + g_num] = -tc.real
            G[j + 3, 1 + g_num] = -tc.imag

#   RHS
        h[j] = det_Hq.real
        h[j + 1] = det_Hq.imag

        h[j + 2] = -det_Hq.real
        h[j + 3] = -det_Hq.imag

        i += 1

#   ?
def fb_check(H_func, q_all, couplings, E_cur, aux):
    """
    How close the Hamiltonian is to having a FB

    Parameters
    ----------
    H_func
    q_all : array
        the wavevectors
    E : float
    """
    dE = _np.zeros_like(q_all)
#    I = _np.eye(Hq.shape[0], dtype=Hq.dtype)

    Hq = aux['Hq']
    I = aux['I']

    i = 0
    for q in q_all:
        Hq[:, :] = H_func(q, couplings) - E_cur*I

        E = _la.eigvalsh(Hq)
        dE[i] = _np.min(_np.abs(E - E_cur))

    return dE

#   ?
def lp_search_1d(H_func, T_all, g_ini, config, p_log=False):
    """
    Search for a flatband solving sequential LP

    Notice that CVXOPT solves the following LP problem
        minimize c@x
        G@x <= h

    Parameters
    ----------
    H_func : function
        function providing the Bloch Hamiltonian matrix
        H_func(q, couplings) should return the Hamiltonian matrix at
        the wavevector q
    T_all : array
        the Hamiltonian matrix decomposition
    g_ini : array
        initial couplings/hopping parameters
    config : dict
        configuration options
    p_log : bool, optional
        show progress log

    Returns
    -------
    g_all : array
        the final couplings/hopping parameters
    loop : int
        number of iterations
    err : float
        convergence error
    E_cur : float
        the flatband energy
    """
#   Parameters
    slp_t = config['slp_type']                                          # SLP type
    err_max = config['err_max']                                         # convergence error
    loop_max = config['loop_max']                                       # maximum number of iterations
    eig_eps = config['eig_eps']                                         # zero eigenvalue threshold
    if 'lp_solver' in config:                                           #
        lp_s = config['lp_solver']                                      # glpk or CVXOPT
    else:                                                               #
        lp_s = None

#   Sanity checks
    assert slp_t in slp_type_all, 'Unsupported type of SLP: ' + slp_t

#   Target FB energy
    no_target = not ('E' in config)                                     # target FB energy not specified

#   RNG
    has_rng = ('rng' in config)                                         #
    if has_rng:
        rng = config['rng']

    nu = T_all.shape[1]                                                 # number of bands

#   Wavevectors
    q_num = 2*nu + 1                                                    # number of q-points
    q_all = _np.zeros(q_num)                                            # the wvavectors
    if has_rng:                                                         # resample the q points on every iteration
        q_gen = lambda: rng.uniform(low=-_mt.pi, high=_mt.pi, size=q_num)
    else:                                                               # use equidistant set of q points
        q_gen = lambda: _np.linspace(-_mt.pi, _mt.pi, q_num, endpoint=False)

#   (Target) FB energy
    E_cur = 0.0                                                         # current approximation to FB energy
    if not no_target:                                                   # the target FB energy specified
        E_cur = config['E']                                             #

#   ?
    g_all = g_ini.copy()                                                # the current set of couplings
    g_all_old = _np.zeros_like(g_all)                                   # the old set of couplings

#   Prepare
    (cf, G, h, aux) = slp_init(config, T_all, g_all, q_num)             #

    if slp_t == 'lndet':                                                #
        constraints_make = lambda: constraints_lndet_make(G, h, H_func, q_all, g_all, E_cur, aux)
    elif slp_t == 'det':                                                #
        constraints_make = lambda: constraints_det_make(G, h, H_func, q_all, g_all, E_cur, aux)

#   ?
    err = 1.0                                                           # safe initial value
    loop = 0                                                            # number of iterations

#   ?
    while err > err_max and loop < loop_max:                            # iterate untile converges or maximum number of iterations exceeded
        q_all[:] = q_gen()                                              # generate set of k points for the current iteration

        constraints_make()                                              # update the constraints

        sol = _co.solvers.lp(cf, G, h, solver=lp_s)                     # solve the LP instance

        if sol['status'] != 'optimal':                                  # found solution
            for key in sol:
                if isinstance(sol[key], _co.matrix):
                    continue

                print('{} = {}'.format(key, sol[key]))

            raise AssertionError('Solving LP failed: status = ' + sol['status'])

        sx = _np.array(sol['x']).squeeze()                              # convert solution to NumPy array - flattened dxd matrix

        g_all_old[:] = g_all                                            # memorise the old couplings

        (E, err) = slp_update(sx, E_cur, g_all, g_all_old, aux)         #
        if no_target:
            E_cur = E

#   Progress log
        if p_log:                                                       #
            dE = fb_check(H_func, q_all, g_all, E_cur, aux)

            print('loop = {}\terr = {:e}\tE = {:e}\tvar(E) = {:e}'.format(loop, err, E_cur, dE.std()))

#   Debug
            G_all = hopping_collect(T_all, config['m_all'], g_all)
            _np.save('loop-{}.npy'.format(loop), G_all, allow_pickle=False)

        loop += 1                                                       #

#   Finalisation
    slp_finalise()                                                      #

    return (g_all, loop, err, E_cur)

#   SLP based search for FB
def hopping_expand(H_all, eps):
    """
    """
    H_all[0] = _np.tril(H_all[0])                                       # lower triangular part of the intracell hopping
    dt = H_all.dtype
    if _np.all(H_all.imag < eps):
        dt = _np.float64

    mc = H_all.shape[0]                                                 # hopping range
    nu = H_all.shape[1]                                                 # number of bands
    g_num = _np.sum(_np.abs(H_all) > eps)                               # number of couplings == number of non-zero elements

    T_all = _np.zeros((g_num, nu, nu), dtype=_np.uint8)                 # the series in couplings = hoppings
    m_all = _np.zeros(g_num, dtype=_np.uint16)                          # hopping range: 0, 1, 2, ... - multiplier in the exponent
    g_ini = _np.zeros(g_num, dtype=dt)                                  # initial hoppings

#   ?
    j = 0
    for i in range(mc):                                                 # loop over the hopping range
        H = H_all[i]                                                    # the current hopping matrix
        (a, b) = _np.where(_np.abs(H) > eps)                            # non-zero hoppings

        for x, y in zip(a, b):                                          # loop over ?
            g_ini[j] = H[x, y]                                          #

            T = _np.zeros((nu, nu), dtype=_np.uint8)
            T[x, y] = 1
            T_all[j, :, :] = T
            m_all[j] = i

            j += 1

    return (T_all, m_all, g_ini)

def hopping_collect(T_all, m_all, g_all):
    """
    Reconstruct hopping matrices from T_all and the couplings g_all

    Parameters
    ----------

    Returns
    -------
    """
    mc = _np.max(m_all)                                                 # hopping range
    nu = T_all.shape[1]                                                 # number of bands
    g_num = len(g_all)                                                  # number of couplings

    G_all = _np.zeros((1 + mc, nu, nu), dtype=g_all.dtype)

#   ?
    j = 0
    while m_all[j] == 0:
        if _np.all(T_all[j].T == T_all[j]):
            G_all[0, :, :] += T_all[j, :, :]*g_all[j]
        else:
            G_all[0, :, :] += (_np.transpose(T_all[j, :, :]) + T_all[j, :, :])*g_all[j]

        j += 1

    for i in range(1, 1 + mc):                                          # loop over the hopping range
        while j < g_num and m_all[j] == i:                              #
            G_all[i, :, :] += T_all[j, :, :]*coupling[j]

            j += 1

    return G_all

def slp_optimise(M, config, eps, p_log=False):
    """

    Parameters
    ----------
    M : QuantumNetwork
    config : dict
    p_log : bool, optional
        progress log

    Returns
    -------
    """
    H_all = M['H_all']                                                  # all the hopping matrices

    (T_all, m_all, g_ini) = hopping_expand(H_all, eps)                  # compute the hopping expansion

    nu = H_all.shape[1]                                                 # number of bands
    g_num = len(g_ini)                                                  # number of couplings

#   Function computing Hamiltonian from the hopping expansion
    if g_ini.dtype == _np.complex128:                                   #
        def H_func(q, g_all):
            """
            ?
            """
            Hq = _np.zeros((nu, nu), dtype=_np.complex128)

            for i in range(g_num):                                      # loop over couplings/hopping parameters
                Ti = T_all[i, :, :]

                Hq += (Ti.T*_cmt.exp(1j*q*m_all[i]) + Ti*_cmt.exp(-1j*q*m_all[i]))*g_all[i]

            return Hq
    else:                                                               # real Bloch Hamiltonian
        def H_func(q, g_all):
            """
            ?
            """
            Hq = _np.zeros((nu, nu), dtype=_np.float64)

            for i in range(g_num):                                      #
                cq = _mt.cos(q*m_all[i])
                sq = _mt.sin(q*m_all[i])
                Ti = T_all[i, :, :]

                Hq += 2*((cq + sq)*Ti + (cq - sq)*Ti.T)

            return Hq

#   Debug
    if p_log:                                                           # progress log
        config['m_all'] = m_all

        G_all = hopping_collect(T_all, config['m_all'], g_ini)
        _np.save('loop-ini.npy', G_all, allow_pickle=False)

#   Optimisation
    (g_all, E, err, loop) = lp_search_1d(H_func, T_all, g_ini, config, p_log=p_log)

#   Finalise: compute the new hopping matrices
    G_all = hopping_collect(T_all, m_all, g_all)

    return (G_all, err, loop, E)
