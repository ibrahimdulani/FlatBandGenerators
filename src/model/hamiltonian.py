#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  hamiltonian.py
#
#  Copyright 2018 Alexei Andreanov <alexei@pcs.ibs.re.kr>
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


#   Bloch Hamiltonian

import math as _mt
import cmath as _cmt

import numpy as _np
import scipy.io as _sio
import scipy.linalg as _la
import scipy.sparse as _ssp

import lsm as _lsm

#   Definitions
#   Internal
_eps_hermiticity = 1e-8                                                 # Hermiticity threshold
_eps_zero = 1e-4                                                        # zero threshold
_hm_key_all = ('M', 'name', 'coupling', 'bd_type', 'bd_var', 'ef_type',
               'ef_var')

#   Public
cmp_eps_default = 1e-8                                                  #

hm_type_all = ('nn', 'nn_so', 'nn2')                                    # Hamiltonian matrix type
hm_ue_method_all = ('ED', 'Pade', 'emm')                                # unitary evolution methods
hm_disorder_type_all = ('dd',)                                          # types of disorder for Hamiltonian matrices

#   Hamiltonian matrices
class Hamiltonian:
    """
    Hamiltoniam matrix
    """
    def __init__(self, HMMatFile, EFMatFile):
        """
        """
        self.load('hm', HMMatFile)                                      # load the Hamiltonian matrix
        if EFMatFile is None:                                           # no external fields
            self.ef_type = ''
            self.ef_var = _np.float64(0.0)
            self.ef_id = _np.int32(-1)
            self.ef_serial = _np.int32(-1)
        else:                                                           # external fields
            self.load('ef', EFMatFile)

#   Hermiticity
        N = len(self)                                                   #

        x = 0.0
#        for i in range(N):
#            nb = self.M.indices[self.M.indptr[i]:self.M.indptr[i+1]]

#            for j in nb:
#                x = max(x, _np.abs(self.M[i, j] - _np.conjugate(self.M[j, i])))
        self.hermiticity = x

#        MT = self.M.T
#        self.hermiticity = _np.max(_np.abs(_np.conjugate(MT.data) - self.M.data))

#   Sanity checks
        assert self.hermiticity < _eps_hermiticity, 'Hamiltonian is not Hermitian!'

    def __len__(self):
        """
        """
        return self.M.shape[0]

    def __getitem__(self, key):
        """
        """
        if key in _hm_key_all:
            key_val = getattr(self, key)
        else:
            raise KeyError('Wrong key ' + key)

        return key_val

#    def __setitem__(self, key, key_val)

    def __eq__(self, H, eps=cmp_eps_default):
        """
        """
        M0 = self.M
        M1 = H['M']

        N = len(self)

        eq = True

        if N != len(H):                                                 # compare Hamiltonian sizes
            eq = False
        elif self.M.data.size != H['M'].data.size:                      # compare?
            eq = False
        else:                                                           # elementwise comparison
            for i in range(N):
                nb0 = M0.indices[M0.indptr[i]:M0.indptr[i+1]]
                nb1 = M1.indices[M1.indptr[i]:M1.indptr[i+1]]

                if nb0.size != nb1.size:
                    eq = False

                    break
                if _np.any(nb0 != nb1):
                    eq = False

                    break
                elif any([_np.abs(M0[i, nb0[j]] - M1[i, nb1[j]]) > eps for j in range(nb0.size)]):
                    eq = False

                    break

        return eq

#   ?
    def name_id(self):
        """
        """
        s = self.name                                                   # name
        try:                                                            # array of couplings
            s += '-' + '-'.join(map(str, self.coupling))
        except TypeError:                                               # single coupling
            s += '-' + str(self.coupling)
        if self.bd_type != 'none':                                      # bond disorder
            s += '-bd-{}-{}'.format(self.bd_type, self.bd_var)
            if self.bd_id > -1:
                s += '-id' + str(self.bd_id)
            if self.bd_serial > -1:
                s += '-' + str(self.bd_serial)
        if self.ef_type == 'uniform':                                   #
            s += '-ef-' + '-'.join(map(str, self.ef_uf))
        if self.ef_type not in ('none', 'uniform'):                     # random fields
            s += '-ef-{}-{}-id{}'.format(self.ef_type, self.ef_var, self.ef_id)
            if self.ef_serial > -1:
                s += '-' + str(self.ef_serial)

        return s

#   Properties
    def is_hermitian(self, eps=1e-8):
        """
        """
        return (self.hermiticity < eps)

#   I/O
    def load(self, fmt, fn):
        """
        """
        data = _sio.loadmat(fn, squeeze_me=True)                        #

        if fmt == 'hm':                                                 # Hamiltonian matrix
            self.name = data['name']
            self.M = _ssp.csr_matrix(data['M'])
            self.coupling = _np.array(data['coupling'], dtype=_np.float64)

            self.bd_type = str(data['bd_type'])
            self.bd_var = _np.float64(data['bd_var'])
            self.bd_id = _np.int32(data['bd_id'])
            self.bd_serial = _np.int32(data['bd_serial'])
        elif fmt == 'ef':                                               # external fields
            ef = data['ef']                                             #

            M = self.M.tolil()                                          #
            N = M.shape[0]                                              #

#   Sanity check
            assert N == len(ef), 'Size of the Hamiltonian matrix is different from the size of the external fields!'

#   ?
            for i in range(N):
                M[i, i] = ef[i]

            self.M = M.tocrs()                                          #

#   ?
            self.ef_type = str(data['ef_type'])
            self.ef_var = _np.float64(data['ef_var'])
            self.ef_id = _np.int32(data['ef_id'])
            self.ef_serial = _np.int32(data['ef_serial'])
            if 'ef_uf' in data:                                         # uniform fields
                self.ef_uf = data['ef_uf']
        else:
            raise ValueError('Loading {} is unsupported!'.format(fmt))

#   Bloch representation
class BlochHamiltonian:
    """
    The Bloch Hamiltonian
    """
    def __init__(self):
        """
        """
        pass

    def bands(self, q_all, g_all):
        """
        Compute the bandsstructure

        Parameters
        ----------
        q_all : array
            the wavevectors
        g_all : array
            the couplings

        Returns
        -------
        Eq : array
            the bandstructure
        """
        q_num = q_all.shape[0]                                          #

        Eq = _np.zeros((q_num, self.nu), dtype=_np.float64)

        for i in range(q_num):                                          # loop over the wavevectors
            Hq = self.bloch_matrix(q_all[i], g_all)                     # compute the Hamiltonian matrix at q_all[i]

            Eq[i, :] = _la.eigvalsh(Hq)                                 #

        return Eq.T

#   ?
def disorder_default(hmd, fmt):
    """
    """
    assert fmt in ('bd', 'ef'), 'Unknown format: ' + fmt

    hmd[fmt + '_id'] = _np.int32(-1)
    hmd[fmt + '_serial'] = _np.int32(-1)

#   Hamiltonian matrix generators
def hm_nn(L, t0):
    """
    n.n. hopping matrix

    Parameters
    ----------
    L : lsm.lattice.Lattice
    t0 : array or complex

    Returns
    -------
    M : sparse array
        the Hamiltonian matrix
    """
    try:
        t_num = len(t0)                                                 # array
    except TypeError:                                                   # number
        t_num = 0

    A = L['A']                                                          # adjacency matrix

    M = _ssp.csr_matrix(A, dtype=_np.complex128)                        #

    if t_num > 1:
        raise NotImplementedError('Has not been implemented yet!')

        sl_num = L['sl_num']                                            #

        assert len(t0) == L['sl_num'], 'Number of hoppings mismatches number of sublattices!'

        sl = L['sl']
    else:                                                               #
        M.data[:] = -t0

    return M

def hm_nn2(L, t0, z2=None, eps=1e-4):
    """
    n.n. and 2nd n.n. hoppings

    Parameters
    ----------
    L : lsm.Lattice
    t0 : array
        the hoppings

    Returns
    -------
    M : sparse array
    """
    assert len(t0) > 1, 'Too few input hoppings!'

#   ?
    N = len(L)                                                          # number of sites

#   n.n.
    M1 = hm_nn(L, t0[0])                                                # n.n. hopping matrix
    M12 = M1.tolil()

#   2nd n.n.
    A2 = _lsm.lattice.generic.adjacency_nnk(L, k=2, zk=z2, eps=eps)

    for i in range(N):
        nb2 = A2.indices[A2.indptr[i]:A2.indptr[i+1]]
        M12[i, nb2] = -t0[1]

    return M12.tocsr()

def hm_nn_so(L, t0, m=1):
    """
    Generate hopping matrix for SO/TE-TM splitting n.n. hopping on a lattice
    I assume the reference axis for phases is the X axis
    Conventions:
        2*i         spin down component
        2*i + 1     spin up component

    Parameters
    ----------
    L : lsm.lattice.Lattice
    t0 : array
        the hoppings:
            0 - without spin flip
            1 - with spin flip
    m : int, optional
        the multiplier of the phase
            1 - SO
            2 - TE/TM

    Returns
    -------
    M : sparse array
        the Hamiltonian matrix
    """
    assert m in (1, 2), 'Invalid value for m!'

#   ?
    N = len(L)                                                          # number of sites
    d = L['d']                                                          # dimension of space, 2 in this case
    A = L['A']                                                          # adjacency matrix

    t = t0[0]                                                           # hopping with no spin-flip
    s = t0[1]                                                           # hopping with a spin-flip

#   ?
    M = _ssp.lil_matrix((2*N, 2*N), dtype=_np.complex128)               # i = 2*j + a, a = 0: spin down, 1: spin up // was: up, down

    bv = _np.zeros(d)                                                   # bond vector
    z = 1j
    mf = (-1)**m

    for i in range(N):                                                  # loop over sites
        nb = A.indices[A.indptr[i]:A.indptr[i+1]]                       # n.n. neighbours of site "i"

        for j in nb:                                                    # loop over neighbours
#   Non-spin flipping part
            M[2*i, 2*j] = -t                                            # spin: down - down
            M[2*j, 2*i] = -t

            M[2*i + 1, 2*j + 1] = -t                                    # spin: up - up
            M[2*j + 1, 2*i + 1] = -t

#   Spin-flipping part
            bv[:] = L.bond_vector(j, i)                                 # bond vector pointing from i to j
            phi = _mt.atan2(bv[1], bv[0])                               # phase
            z = _cmt.exp(-1j*m*phi)                                     #

            M[2*j + 1, 2*i] = -s*z                                      # i, spin down --> j, spin up
            M[2*i, 2*j + 1] = -s*z.conjugate()                          # j, spin up --> i, spin down

            M[2*j, 2*i + 1] = -mf*s*z.conjugate()                       # i, spin up --> j, spin down
            M[2*i + 1, 2*j] = -mf*s*z                                   # j, spin down --> i, spin up

    return M.tocsr()

def hm_make(L, fmt, t0, config, eps=1e-8):
    """
    Universal Hamiltonian matrix generator

    Parameters
    ----------
    L : lsm.lattice.Lattice
    fmt : str
        matrix type
    t0 : array of float
        hopping(s)
    config : dict or None
        configuration
    eps : float
        zerho threshold

    Returns
    -------
    M : sparse array
    """
    if fmt == 'nn':                                                     # n.n. hopping
        M = hm_nn(L, t0)                                                #
    elif fmt == 'nn2':                                                  # n.n. + 2nd n.n. hoppings
        M = hm_nn2(L, t0, z2=config['z2'])
    elif fmt == 'nn_so':                                                # n.n. hopping with spin flips
        M = hm_nn_so(L, t0, m=config['m'])
    else:
        raise ValueError('Uknown Hamiltonian matrix type: ' + fmt)

#   Sanity checks
#    MT = M.T
#    assert _np.all(_np.abs(_np.conjugate(MT.data) - M.data) < eps), 'Hamiltonian matrix is not Hermitian!'
#    assert _np.all(_np.abs(_np.conjugate(M.T) - M) < eps), 'Hamiltonian matrix is not Hermitian!'

#   Unitarity check
    N = M.shape[0]

    for i in range(N):
        nb = M.indices[M.indptr[i]:M.indptr[i+1]]

        assert all([_np.abs(_np.conjugate(M[j, i]) - M[i, j]) < eps for j in nb]), 'Hamiltonian matrix is not unitary: row = ' + str(i)

#   Hermitian conjugate of M
#    MT = M.T
#    MT.data[:] = _np.conjugate(MT.data)

#    dM = MT - M

#    assert _np.all(_np.abs(dM.data) < eps), 'Hamiltonian matrix is not unitary!'

    return M

#   ?
def hm_disorder_make(M, fmt, config):
    """
    Generate disordered Hamiltonian

    Parameters
    ----------
    M : sparse array
        the Hamiltonian matrix
    fmt : str
        type of disorder
    config : dict

    Returns
    -------
    Md : sparse array
    """
    if fmt == 'dd':                                                     # diagonal disorder
        rng = config['rng']                                             # RNG
        dd_type = config['disorder_type']                               #
        dd_var = config['disorder_var']                                 #

        N = M.shape[0]                                                  # number of sites

        if dd_type == 'gauss':                                          # Gaussian distribution
            dd = rng.normal(scale=dd_var, size=N)
        elif dd_type == 'box':                                          # box distribution
            dd = rng.uniform(low=-0.5*dd_var, high=0.5*dd_var, size=N)
        else:
            raise ValueError('Unknown disorder distribution type: ' + dd_type)

        Md = _ssp.lil_matrix(M)                                         #
        for i in range(N):
            Md[i, i] = dd[i]
    else:
        raise ValueError('Unknown disorder type: ' + fmt)

    return Md.tocsr()

#   ?
def hm_submatrix(M, E, ix, eps=_eps_zero, sparse=False):
    """
    Cut out a submatrix from the Hamiltonian matrix M given by subset ix

    Parameters
    ----------
    M : sprase array
        the Hamiltonian matrix
    E : float
        the shift
    ix : array
        the subset
    eps : float, optional
        the zero treshold
    sparse : bool, optional
        generate sparse submatrix

    Returns
    -------
    M_ix : sparse array
        the sub-Hamiltonian matrix
    """
    if sparse:                                                          #
        M_ix = _ssp.lil_matrix((ix.size, ix.size), dtype=M.dtype)       #
    else:
        M_ix = _np.zeros((ix.size, ix.size), dtype=M.dtype)

    ix_s = _np.sort(ix)                                                 #
    ix_r = {ix_s[i]:i for i in range(ix.size)}                          #

    for i in range(ix.size):                                            # loop over subset (size)
        r = M.indices[M.indptr[ix[i]]:M.indptr[ix[i]+1]]                # non-zero elements in ix[i]'th row of M
        r = [j for j in r if j in ix]                                   #
        p = [ix_r[j] for j in r]                                        #
        M_ix[i, p] = M.data[r]
        M_ix[i, i] -= E

    if sparse:
        M_ix = M_ix.tocsr()

    return M_ix

#   ?
def fb_monte_carlo(M, config):
    """
    Flatband Monte-Carlo

    Parameters
    ----------
    M : BlochHamiltonian
    config : dict
        Monte-Carlo configuration

    Returns
    -------
    g_all : array
        the couplings
    """
    g_all = config['g_all']                                             #
    rng = config['rng']

    return g_all

#   Unitary evolution
def unitary_evolution(t_all, M, psi0, method='ED', sp=None):
    """
    unitary evolution of initial wavefunction psi0 with Hamiltonian matrix
    M for a discrete set of times t_all

    Parameters
    ----------
    t_all : array
        discrete set of times
    M : sparse array
    psi0 : array
        initial wavefunction
    method : str, optional
        how the evolution operator U is computed
            'ED'   - exact diagnalisation of the Hamiltonian matrix
            'Pade' - using Pade approximants for the evolution operator
                     Quite slow
            'emm'  - using scale and square algorithm, fast but inaccurate
                     for reasons I do not understand
    sp : dict or None, optional
        precomputed spectrum of the Hamiltonian matrix, only used by the
        ED method

    Returns
    -------
    psi : array
        the evolved wavefunctions
    """
#   Sanity checks
#    assert method in hm_ue_method_all, 'Unsupported unitary evolution method: ' + method
    assert M.shape[0] == M.shape[1], 'Hamiltonian matrix is not rectangular!'
    if method == 'ED':
        assert M.shape[0] < 2048, 'Hamiltonian matrix is too large for the current naive unitary evolution!'
    assert M.shape[0] == psi0.size, 'Hamiltonian matrix and initial wavefunctions mismatch!'

#   ?
    psi = _np.zeros((t_all.size, psi0.size), dtype=_np.complex128)
    t_num = t_all.size                                                  #

    if method == 'ED':                                                  # exact diagonalisation
        if sp is None:                                                  # no precomputed spectrum provided
            if _ssp.issparse(M):
                M_dense = M.todense()
            else:
                M_dense = M

            (E, phi) = _la.eigh(M_dense)                                # diagonalise the Hamiltonian matrix: E, |phi>
        else:                                                           # use precomputed spectrum
            E = sp['E']
            phi = sp['psi']

        w = psi0@phi.conjugate()                                        # w_a = <phi_a|psi0>
        z = _np.zeros_like(E, dtype=_np.complex128)                     #

        for i in range(t_num):
            z[:] = _np.exp(-1j*E*t_all[i])*w
            psi[i, :] = phi@z
    elif method == 'Pade':                                              # Pade approximants
#        U = _np.zeros(M.shape, dtype=_np.complex128)
        M1 = M.tocsc()

        for i in range(t_num):
#            U[:, :] = _la.expm(-1j*t_all[i]*M)
            U = _la.expm(-1j*t_all[i]*M1)
            psi[i, :] = U@psi0
    elif method == 'emm':                                               # scale and square
        assert _np.std(_np.diff(t_all)) < 1e-6, 'Input t_all is not a linspace-like!'

        psi[:, :] = _ssp.linalg.expm_multiply(-1j*M, psi0, start=t_all.min(), stop=t_all.max(), num=t_all.size, endpoint=True)
    else:
        raise ValueError('Unsupported method: ' + method)

    return psi
