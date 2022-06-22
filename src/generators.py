#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  generators.py
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


#   Various flatband generators

import math as _mt
import cmath as _cmt

import numpy as _np
import scipy.linalg as _la
import scipy.optimize as _sop

#   Definitions
#   Internal

#   Public
fmt_d1mc1U2_all = ('cs', 'stub', 'diamond', 'lieb', 'fk')               #
fmt_d1mc1_all = ('nh_ge',)                                              #
fmt_random_all = ('random', 'nu2', 'U2')                                #
fmt_all = fmt_d1mc1U2_all + fmt_d1mc1_all + fmt_random_all

#   ?
def block_matrix(nu, mu, mask, rng):
    """
    Generate block matrix specified by the mask - 4 element vector
    [a, b, c, d] --> a b    0 1
                     c d    2 3
    b == -c implies c = b.T

    Parameters
    ----------
    nu : int
        number of bands
    mu : int
        size of the majority sublattice
    mask : array or None
        the blocks to be set to zero
    rng : numpy.RandomState
        the RNG

    Returns
    -------
    M : array
        the matrix
    """
    M = _np.zeros((nu, nu))                                             #

    if mask[0] != 0:                                                    #
        M[:mu, :mu] = rng.normal(size=(mu, mu))
    if mask[1] != 0:
        A = rng.normal(size=(nu-mu, mu))                                # the block

        M[:mu, mu:] = A.T
        if mask[1] == -mask[2]:                                         # bipartite H0
            M[mu:, :mu] = A
    if mask[2] != 0 and mask[1] != -mask[2]:                            #
        M[mu:, :mu] = rng.normal(size=(nu-mu, mu))
    if mask[3] != 0:                                                    #
        M[mu:, mu:] = rng.normal(size=(nu-mu, nu-mu))

#    M = _np.bmat([[_np.zeros((mu, mu)), A.T], [A, _np.zeros((nu-mu, nu-mu))]])
#    M = _np.block([[_np.zeros((mu, mu)), A.T], [A, _np.zeros((nu-mu, nu-mu))]])

    return M

#   Intracell hopping
def H0_make(config):
    """
    Generate intracell hopping matrix
    Only non-degenerate construction has been implemented so far

    Parameters
    ----------
    config : dict
        the configuration

    Returns
    -------
    H0 : array
    """
#   Parameters
    nu = config['nu']                                                   # number of bands
    rng = config['rng']                                                 # RNG

#   ?
    if 'bipartite' in config:                                           # bipartite lattice
        mu = config['mu']                                               # number of sites in the unit cell on the majority sublattice

        H0 = block_matrix(nu, mu, (0, 1, -1, 0), rng)                   #
    else:                                                               # non-bipartite lattice
        if 'mp' in config:                                              #
            mp = config['mp']                                           # multiplicities
        else:                                                           #
            mp = None

#   ?
        t = _np.zeros(nu)                                               # diagonal elements of H0
        if mp is None:                                                  # trivial multiplicities
            t[1] = 1.
            if nu > 2:
                assert rng is not None, 'More than two bands and no RNG!'

                t[2:] = 1. + _np.sort(rng.uniform(size=nu-2))           #
        else:                                                           #
            assert sum(mp) == nu, 'Sum of multiplicities mismatch the number of bands!'

            mp_pum = len(mp)                                            # number of distinct diagonal elements
            mp_cs = _np.cumsum(mp)                                      # cumulative sums of multiplicities
            t[:mp_cs[0]] = 0.
            t[mp_cs[0]:mp_cs[1]] = 1.
            if mp_num > 2:                                              #
                s = rng.uniform(size=mp_num - 2)                        #

                for i in range(1, mp_num-1):
                    t[mp_cs[i]:mp_cs[i+1]] = s[i - 1]

        H0 = _np.diag(t)                                                #

    return H0

#   n.n. hopping
def _nu2_make(config):
    """
    Generate nu=2 flatband H1

    Parameters
    ----------
    config : dict

    Returns
    -------
    H1 : array
    """
    theta = config['theta']
    phi = config['phi']
    alpha = config['alpha']                                             # phase

    g = _mt.sqrt(-_mt.sin(2*theta)*_mt.sin(2*phi))/abs(_mt.sin(2*(theta - phi)))
    H1 = _np.array([[_mt.cos(theta)*_mt.cos(phi), _cmt.exp(-1j*alpha)*_mt.cos(theta)*_mt.sin(phi)],
                    [_cmt.exp(1j*alpha)*_mt.sin(theta)*_mt.cos(phi), _mt.sin(theta)*_mt.sin(phi)]])

    return g*H1

#   U=2 with arbitrary number of bands
def _U2_make_bipartite(H0, config):
    """
    Construct U=2 n.n. hopping matrix for any number of bands
    Bipartite case

    Parameters
    ----------
    H0 : array
        intracell hopping
    config : dict

    Returns
    -------
    H1 : array
    """

#   Parameters
    rng = config['rng']                                                 # RNG
    nu = H0.shape[0]                                                    # number of bands
    mu = config['mu']                                                   # number of sites in the unit cell on the majority sublattice
    I = _np.eye(mu)                                                     # identity matrix

#   Sanity checks
    assert 'bipartite' in config, 'Non-bipartite network!'
    assert 'type' in config, 'No type of bipartitness is supplied!'
    assert nu - mu == mu - 1, 'Only disbalance = 1 has been implemented for bipartite lattices!'
    assert _np.allclose(H0[:mu, :mu], 0.) and _np.allclose(H0[mu:, mu:], 0.), 'H0 does not have the bipartite structure!'

#   ?
    A = H0[mu:, :mu]                                                    #

#   Righ zero eigenvector of H1, majority sublattice
    if 'phi1' in config:                                                #
        phi1 = _np.array(config['phi1'])
    elif config['type'] == 'upper':                                     # only upper right block of H1 is non-zero
        phi1 = _np.ones(mu)
        phi1[:mu-1] = _la.solve(A[:, :-1], -_np.ravel(A[:, -1]))        # kernel?
    else:                                                               #
        phi1 = rng.normal(size=mu)

#   Left zero eigenvector of H1, majority sublattice
    if 'phi2' in config:                                                #
        phi2 = _np.array(config['phi2'])
    elif config['type'] == 'lower':                                     # only lower left block of H1 is non-zero
        phi2 = _np.ones(mu)
        phi2[:mu-1] = _la.solve(A[:, :-1], -_np.ravel(A[:, -1]))        # kernel?
    else:                                                               #
        phi2 = rng.normal(size=mu)

#   Norms
    nphi = _np.array([[phi1@phi1, phi1@phi2], [phi2@phi1, phi2@phi2]])
    K = _np.array([[1., nphi[1, 0]/nphi[1, 1]], [nphi[0, 1]/nphi[0, 0], 1.]])
    ab = _la.solve(K, _np.ones(2))                                      #

#   Generate the blocks of H1
    H1 = _np.zeros((nu, nu))                                            #
    if config['type'] != 'upper':                                       # lower left block of H1 is non-zero
        Q = I - _np.outer(phi1, phi1)/(phi1@phi1)                       # transverse projector on phi2

        v = Q@rng.normal(size=mu)                                       #
        S = -_np.outer(A@phi1, v)/(v@phi2)                              # particular solution
        if particular:                                                  #
            dS = _np.zeros((mu, mu))
        else:                                                           #
            dS = rng.normal(size=(nu-mu, mu))@(I - ab[0]*_np.outer(phi1, phi1)/nphi[0, 0] - ab[1]*_np.outer(phi2, phi2)/nphi[1, 1])

        H1[mu:, :mu] = S + dS
    if config['type'] != 'lower':                                       # upperf right block of H1 is non-zero
        Q = I - _np.outer(phi2, phi2)/(phi2@phi2)                       # transverse projector on phi2

        v = Q@rng.normal(size=mu)                                       #
        T = -_np.outer(A@phi2, v)/(v@phi1)                              # particular solution
        if particular:                                                  #
            dT = _np.zeros((mu, mu))
        else:                                                           #
            dT = rng.normal(size=(nu-mu, mu))@(I - ab[0]*_np.outer(phi1, phi1)/nphi[0, 0] - ab[1]*_np.outer(phi2, phi2)/nphi[1, 1])

        H1[:mu, mu:] = _np.transpose(T + dT)

    return H1

def _U2_psi_resolve(H0, psi1, psi2):
    """
    Find compatible Psi_2|Psi1

    Parameters
    ----------
    H0 : array
        intracell hoppin
    psi1 : array
        right zero eigenvector

    Returns
    -------
    psi2 : array
        left zero eigenvector
    """
    a = psi1@H0@psi1                                                    #
    npsi1 = psi1@psi1                                                   #
    t0 = H0.diagonal()                                                  # diagonal elements of H0

#        f = lambda x: (psi1@x - 1.0, _np.sum(eps*x**2 + eps*psi1*x*(npsi1 - x@x)) - a, 0.)
    def psi2_func(x):
        y = _np.zeros_like(x)
        y[0] = psi1@x - 1.0
        y[1] = _np.sum(t0*x**2 + t0*psi1*x*(npsi1 - x@x)) - a

        return y
#        df = lambda x: (psi1, 2*eps*x + eps*psi1*(npsi1 - x@x) - 2*_np.sum(eps*psi1*x)*x, _np.zeros(nu))
    def psi2_dfunc(x):
        nu = x.size
        dy = _np.zeros((nu, nu))
        dy[0, :] = psi1
        dy[1, :] = 2*(t0*x + t0*psi1*(npsi1 - x@x) - _np.sum(t0*psi1*x)*x)

        return dy

    t = psi1@psi2
    psi2 /= t
#        res = _sop.root(f, psi2, jac=df, method='hybr')                # find psi2
    res = _sop.root(psi2_func, psi2, jac=psi2_dfunc, method='hybr')     # find psi2

#   Sanity checks
    assert res['success'], 'Failed to find psi2: {}'.format(res)
#        assert _np.all(res['fun'] < eps), 'Failed to find the root for psi2!'

#   The numerical solution
    psi2 = res['x']                                                     # left zero mode of H1

    return psi2

def _U2_psi12_make(H0, config):
    """
    Get the compatible set of psi1, psi2

    Parameters
    ----------
    H0 : array
        intracell hopping
    config : dict

    Returns
    -------
    psi1, psi2 : array
        the left and right eigenmodes
    """
#   Parameters
    nu = config['nu']                                                   # number of bands
    rng = config['rng']                                                 # RNG
    if 'E' in config:                                                   # supplied FB energy
        E = config['E']                                                 # flatband energy
    else:
        E = rng.normal(scale=H0.max())

#   ?
    if 'psi1' in config:                                                # right zero mode supplied
        psi1 = _np.array(config['psi1'])                                # right zero mode of H1
    else:                                                               #
        psi1 = rng.normal(size=nu)                                      # right zero mode of H1

#   ?
    if 'psi2' in config:                                                # precomputed left zero mode
        psi2 = config['psi2']                                           # precomputed left zero mode of H1
    else:                                                               # no precomputed left zero mode of H1 provided, find it then
        psi2 = rng.uniform(size=nu)                                     #

#   ?
    if 'psi1' in config and 'psi2' in config:                           # both eigenmodes are predefined
        pass
    elif 'psi1' in config and 'psi2' not in config:                     # only psi1 is predefined
        psi2 = _U2_psi_resolve(H0, psi1, psi2)                          #
    elif 'psi1' not in config and 'psi2' in config:                     # only psi2 is predefined
        psi1 = _U2_psi_resolve(H0, psi2, psi1)
    else:                                                               # neither psi1 or psi2 are predefined
        psi2 = _U2_psi_resolve(H0, psi1, psi2)                          # priority goes to psi1, and psi2 is constructed

#   Sanity check
    assert _np.allclose(psi2@psi1, 1.0), '<psi2|psi1> != 1!'

    return (psi1, psi2)

def _U2_make(H0, config):
    """
    Construct U=2 n.n. hopping matrix for any number of bands

    Parameters
    ----------
    H0 : array
        intracell hopping
    config : dict

    Returns
    -------
    H1 : array
    """
#   Parameters
    eps = 1e-10                                                         # zero threshold
    rng = config['rng']                                                 # RNG
    nu = H0.shape[0]                                                    # number of bands
    particular = config['particular']                                   # generate particular solution only?
    I = _np.eye(nu)                                                     # unit matrix

#   Sanity checks
    assert nu > 1, 'At least two bands are needed to construct a flatband!'

#   ?
    if 'bipartite' in config:                                           # construct chiral flatband
        H1 = _U2_make_bipartite(H0, config)
    else:                                                               # generic, non-chiral flatband
        (psi1, psi2) = _U2_psi12_make(H0, config)                       # construct valid psi1 and psi2
        npsi1 = psi1@psi1
        npsi2 = psi2@psi2

#   Projectors
        Q1 = I - _np.outer(psi1, psi1)/npsi1
        Q2 = I - _np.outer(psi2, psi2)/npsi2

        Q21 = I - _np.outer(Q2@psi1, Q2@psi1)/(psi1@Q2@psi1)
        Q12 = I - _np.outer(Q1@psi2, Q1@psi2)/(psi2@Q1@psi2)

#   ?
        E = psi2@H0@psi1                                                # flatband energy
        EH0 = E*I - H0                                                  #
        x = psi1@EH0@psi1                                               #

#   Sanity check
        assert _np.allclose(x, psi2@EH0@psi2), 'Incorrect psi2!'

#   Compute the "free" part
        if particular or nu == 2:                                       # particular/basic solution, no "free" part
            K = _np.zeros((nu, nu))
        elif nu > 2:                                                    # free part is there
            if 'complex' in config:                                     # complex "free" part
                r = rng.uniform(size=(nu, nu))                          # the "free" part
                phi = rng.uniform(low=-_mt.pi, high=_mt.pi, size=(nu, nu))
                K = r*_np.exp(1j*phi)
            else:                                                       # real "free" part
                K = rng.uniform(size=(nu, nu))                          # the "free" part
                K = Q21@K@Q12

#   ?
        T = _np.outer(EH0@psi1, EH0@psi2)/x                             # basic solution
        H1 = Q2@(T + K)@Q1                                              # the hopping matrix

#   Sanity checks
        assert _np.allclose(K@Q1@psi2, 0.), 'K0 violated!'
        assert _np.allclose(psi1@Q2@K, 0.), 'K1 violated!'
        assert _np.allclose(T@Q1@psi2, EH0@psi1), 'T0 violated!'
        assert _np.allclose(psi1@Q2@T, psi2@EH0), 'T1 violated!'
        assert _np.allclose(H1@psi2, EH0@psi1), 'First equation is violated!'
        assert _np.allclose(psi1@H1, psi2@EH0), 'Second equation is violated!'

        assert _np.allclose(H0@psi1 + H1@psi2, E*psi1), 'Eig0 violated!'
        assert _np.allclose(H1.T@psi1 + H0@psi2, E*psi2), 'Eig1 violated!'
        assert _np.allclose(H1@psi1, 0.), 'No leak 0 violated!'
        assert _np.allclose(psi2@H1, 0.), 'No leak 1 violated!'

        if not particular:                                              # non-particular/basic solution
            (t, l, r) = _la.eig(H1, left=True)
            ix = _np.where(_np.abs(t) < eps)[0]

            assert ix.size == 1, 'There should be a single zero mode!'
#    assert
#    print('E = {:e}\npsi1 = {}\npsi2 = {}'.format(E, psi1, psi2))

    return H1

#   U=3 case, arbitrary number of bands
def _U3_make(H0, config):
    """
    Generate U=3 flatband model

    Parameters
    ----------
    H0 : array
        intracell hopping
    config : dict
        the configuration

    Returns
    -------
    H1 : array
        the n.n. intercell hopping
    """
#   Generate Psi1, Psi3
    (psi1, psi3) = _U3_psi13_make(H0, config)

#   Generate Psi2
    psi2 = _U3_psi2_make(H0, config, psi1, psi3)

#   Generate H1_base
    psi = _np.array([psi1, psi2, psi3])
    H1_base = _U3_H1_base_make(H0, config, psi)

#   Generate H1_free
    if 'free' in config:                                                # generate the free part
        H1_free = _U3_H1_free_make(H0, config, psi)

    return H1_base + H1_free

#   U>3 with arbitrary number of bads
def _U_psi_make(H0, config):
    """
    Compute compatible set of psi1,.., psiU
    """
    psi_all = _np.zeros(config['U'], config['nu'])

    return psi_all

def _U_T_make(psi_all):
    """
    Construct the T matrix
    """
#   Parameters
    (U, nu) = psi_all.shape

    T = _np.zeros(((U+2)*nu, nu**2))

    return T

def _U_make(H0, config):
    """
    Generate intercell hopping matrix H1

    Parameters
    ----------
    H0 : array
        intracell hopping
    config : dict
        the configuration

    Returns
    -------
    H1 : array
    """
    if 'bipartite' in config:                                           #
        raise NotImplementedError('Bipartite networks have not been implemented yet!')
    else:                                                               #
        nu = config['nu']                                               # number of bands
        U = config['U']                                                 # target CLS size
        rng = config['rng']                                             # RNG

        psi_all = _U_psi_make(H0, config)                               # find compatible set of psi1, .. psiU
        E = psi_all[0]@H0@psi_all[U-1]
        T = _U_T_make(psi_all)                                          # generate the matrix T

        rhs = _np.zeros((U+2)*nu)
        rhs[:U*nu] = _np.ravel((E - H0)*psi_all)

        res = _la.lstsq(T, rhs)                                         # least squares based particular solution

        ker = _U_kernel(T)                                              # basis of the right kernel of T
        c = rng.normal(size=(ker.shape[0]))                             #

#   Sanity checks

#   ?
        H1 = res[0]

    return H1

#   ?
def H1_make(H0, config):
    """
    Generate flatband n.n. hopping matrix

    Parameters
    ----------
    H0 : array or None
        intracell hopping matrix
    config : dict
        parameters for the generator

    Returns
    -------
    H1 : array
    """
#   Sanity check
    if 'bipartite' not in config:                                       # non-bipartite lattice/Hamiltonian
        assert _np.allclose(H0, _np.diag(_np.diag(H0))), 'H0 is not diagonal!'

        bipartite = False                                               # non-bipartite
    else:                                                               # bipartite flag present in config
        assert 'mu' in config, 'Missing number of sites in the unit cell on the majority sublattice!'

        bipartite = config['bipartite']                                 # bipartite flag from config
#    if 'chiral' in config:                                              #
#        assert 'mu' i

#   Generation
    fmt = config['fmt']                                                 # hopping matrix type
    if fmt == 'nu2':                                                    # two band case
        H1 = _nu2_make(config)                                          #
    elif fmt == 'U2':                                                   # U=2 case
        H1 = _U2_make(H0, config)
    elif fmt == 'U3':                                                   # U=3 case
        H1 = _U3_make(H0, config)
    elif fmt == 'U':                                                    # arbitrary U > 2
        H1 = _U_make(H0, config)
    elif fmt == 'random':                                               # completely random n.n. hopping
        nu = config['nu']                                               # number of bands
        rng = config['rng']                                             # RNG

        if bipartite:                                                   # bipartite
            mu = config['mu']                                           # number of sites in the unit cell on majority sublattice

            H1 = block_matrix(nu, mu, (0, 1, 0, 0), rng)                #
        elif 'complex' in config:                                       # non-bipartite, complex hoppings
            r = rng.uniform(size=(nu, nu))                              # moduli
            phi = rng.uniform(low=-_mt.pi, high=_mt.pi, size=(nu, nu))  # phases
            H1 = r*_np.exp(1j*phi)
#            H1_re = rng.uniform(size=(nu, nu))                          # real part
#            H1_im = rng.uniform(size=(nu, nu))                          # imaginary part
#            H1 = H1_re + 1j*H1_im
        else:                                                           # non-bipartite, real hoppings
            H1 = rng.normal(size=(nu, nu))
    else:
        raise ValueError('Uknown format ' + fmt)

    return H1

#   Known lattice models
def __hoppings_read(h_num, config):
    """
    Generate the hopping vector

    Parameters
    ----------
    h_num : int
        number of independent hoppings
    config : dict
        configuration dict

    Returns
    -------
    h : array
        the independent hoppings
    """
    h = _np.ones(h_num)                                                 # total number of independent hoppings
    hd_all = {'h' + str(i): i for i in range(h_num)}                    #

    if 'h' in config:                                                   # hopping vector provided
        h[:] = config['h']
    else:                                                               #
        for key in hd_all:                                              #
            if key in config:                                           #
                h[hd_all[key]] = config[key]

    return h

def _cross_stitch_make(config):
    """
    Cross-stitch chain
    """
    h = __hoppings_read(1, config)                                      #

    H0 = _np.array([[0., h[0]], [h[0], 0.]])
    H1 = _np.ones((2, 2))

    return (H0, H1)

def _stub_make(config):
    """
    Stub lattice model

    Parameters
    ----------
    config : dict
    """
    h = __hoppings_read(3, config)                                      #

    H0 = _np.array([[0., h[0], 0.], [h[0], 0., h[1]], [0., h[1], 0.]])
    H1 = _np.array([[0., h[2], 0.], [0., 0., 0.], [0., 0., 0.]])

    return (H0, H1.T)

def _diamond_make(config, eps=1e-4):
    """
    Diamond chain model

     0 - [2] - 1        [1] - 4 - [1]
     |         |             [2]
    [1]       [1]             |
     |         |              5
     3 - [0] - 2              |
                             [0]


    Parameters
    ----------
    config : dict
        configuration
    eps : float, optional
        zero threshold
    """
#   Parameters
    nu = 3                                                              # number of bands
    phi = config['phi']                                                 # phase due to magnetic field
    h = __hoppings_read(6, config)                                      #

#   Intracell hopping
    dt = _np.float64
    if abs(phi) < eps:                                                  # zero phase
        H0 =  _np.array([[0., h[0], h[5]],
                         [h[0], 0., h[1]],
                         [h[5], h[1], 0.]])
    elif abs(abs(phi) - _mt.pi) < eps:                                  # phase is +-pi
        H0 =  _np.array([[0., h[0], h[5]],
                         [h[0], 0., -h[1]],
                         [h[5], -h[1], 0.]])
    else:                                                               # generic, non-zero phase
        dt = _np.complex128

        H0 = _np.array([[0., h[0], h[5]],
                        [h[0], 0., h[1]*_cmt.exp(1j*phi)],
                        [h[5], h[1]*_cmt.exp(-1j*phi), 0.]], dtype=dt)

#   Intercell hopping
    H1 = _np.zeros((nu, nu), dtype=dt)
    H1[1, 0] = h[2]
    H1[1, 1] = h[4]
    H1[1, 2] = h[3]

    return (H0, H1.T)

def _lieb_make(config, eps=1e-8):
    """
    Lieb chain model
    """
#   Parameters
    nu = 5                                                              # number of bands
    phi = config['phi']                                                 # phase due to magnetic field
    h = __hoppings_read(6, config)                                      #
    dt = _np.float64

#   Intracell hopping
    if abs(phi) < eps:                                                  # zero phase
        H0 = _np.diag(h[:4], k=-1) + _np.diag(h[:4], k=1)
    elif abs(abs(phi) - _mt.pi) < eps:                                  # phase is +-pi
        H0 = _np.diag(h[:4], k=-1) + _np.diag(h[:4], k=1)
        H0[0, 1] = -h[0]
        H0[1, 0] = -h[0]
    else:                                                               # generic, non-zero phase
        dt = _np.complex128

        H0 = _np.eye(nu, k=1, dtype=_np.complex128) + _np.eye(nu, k=-1, dtype=_np.complex128)
        H0[0, 1] = h[0]*_cmt.exp(1j*phi)
        H0[1, 0] = h[0]*_cmt.exp(-1j*phi)

#   Intercell hopping
    H1 = _np.zeros((nu, nu), dtype=dt)
    H1[0, 1] = h[4]
    H1[4, 3] = h[5]

    return (H0, H1.T)

def _fk_make(config):
    """
    Fyodor Kusmartsev lattice
    """
    nu = 5                                                              # number of atoms in the unit cell
    alpha = config['alpha']                                             #
    h = __hoppings_read(3, config)                                      # u = h[0], t = h[1], p = h[2]

    if _np.iscomplex(alpha):
        dt = _np.complex128
    else:
        dt = _np.float64

#   Intracell hopping
    H0 = h[1]*(_np.eye(nu, k=-1, dtype=dt) + _np.eye(nu, k=1, dtype=dt))
    H0[2, 2] = h[0]
    H0[0, 1] = alpha*h[2]
    H0[1, 0] = _np.conjugate(alpha)*h[2]
    H0[3, 4] = h[2]
    H0[4, 3] = h[2]

#   Intercell hopping
    H1 = _np.zeros((nu, nu), dtype=dt)
    H1[0, 2] = h[1]
    H1[4, 2] = h[1]

    return (H0, H1)

def _sawtooth_make(config):
    """
    Sawtooth lattice
    """
    nu = 2                                                              # number of atoms in the unit cell/bands
    alpha = config['alpha']                                             #

    if _np.iscomplex(alpha):
        dt = _np.complex128
    else:
        dt = _np.float64

    H0 = _np.array([[0.0, 1.0], [1.0, 1.0]], dtype=dt)                  # intracell hopping
    H1 = _np.array([[alpha, 1.0], [0.0, 0.0]], dtype=dt)                # intercell hopping

    return (H0, H1)

def _nh_ge_make(config):
    """
    non Hermitian lattice proposed by Li Ge in PRL 2018
    """
    nu = 2                                                              # number of atoms in the unit cell/bands
    gamma = config['gamma']                                             #
    t = config['t_nn']                                                  #

    H0 = _np.array([[1j*gamma, t], [t, -1j*gamma]], dtype=_np.complex128)# intracell hopping
    H1 = _np.array([[0.0, t], [t, 0.0]], dtype=_np.complex128)          # intercell hopping

    return (H0, H1)

#   Umbrella function
def H_make(config):
    """
    Generate the Hamiltonian hopping matrices

    Parameters
    ----------
    config : dict
        the configuration

    Returns
    -------
    H_all : array
        the hopping matrices
    H_data : dict, optional
        extra data
    """
#   Sanity checks
    assert config['nu'] > 1, 'At least two bands are required to construct a flatband!'
    assert config['mc'] == 1, 'Only n.n. hopping has been implemented!'

#   ?
    fmt = config['fmt']                                                 # format
    if fmt == 'stub':                                                   # stub chain
        (H0, H1) = _stub_make(config)
    elif fmt == 'cs':                                                   # cross-stitch
        (H0, H1) = _cross_stitch_make(config)
    elif fmt == 'diamond':                                              # diamond chain
        (H0, H1) = _diamond_make(config)
    elif fmt == 'lieb':                                                 # Lieb chain
        (H0, H1) = _lieb_make(config)
    elif fmt == 'fk':                                                   #
        (H0, H1) = _fk_make(config)
    elif fmt == 'nh_ge':                                                #
        (H0, H1) = _nh_ge_make(config)
    elif fmt in fmt_random_all:                                         # random chains
        if 'H0' in config:                                              # precomputed H0
            H0 = config['H0']
        else:                                                           #
            H0 = H0_make(config)                                        # generate intracell hopping
        H1 = H1_make(H0, config)                                        # generate n.n. hopping
    else:
        raise ValueError('Unknown generator: ' + fmt)

#   Sanity check
    assert H0.dtype == H1.dtype, 'Types of H0 and H1 mismatch!'

    return (H0, H1)
