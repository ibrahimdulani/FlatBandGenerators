#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  kagome.py
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


#   Tight-binding models on kagome lattice

import math as _mt
import cmath as _cmt

import numpy as _np
import scipy.sparse as _ssp
from scipy.linalg import norm as _norm

import lsm as _lsm

from .hamiltonian import BlochHamiltonian as _BlochHamiltonian

from . import generic as _gen
from . import wf as _wf

#   Definitions
#   Internal


#   Public
wf_type_all = ('cls_hex',)                                              #

#   ?
class KagomeBlochHamiltonian(_BlochHamiltonian):
    """
    Tight binding Hamiltonian on the kagome lattice
    """
    def __init__(self):
        """
        """
        s3 = 0.5*_mt.sqrt(3)

        self.a = _np.array([[-0.5, -s3],
                            [1.0, 0.0],
                            [-0.5, s3]])

        self.nu = 3                                                     # number of bands

    def bloch_matrix(self, q, t):
        """
        Compute the Hamiltonian matrix at wavevector q and couplings t

        Parameters
        ----------
        q : array
            wavevector
        t : array
            the couplings, 3 for now

        Returns
        -------
        Hq : array
            the Hamiltonian matrix
        """
        tcqa = t*_np.cos(self.a@q)

        Hq = 2*_np.array([[0.0, tcqa[0], tcqa[2]],
                          [tcqa[0], 0.0, tcqa[1]],
                          [tcqa[2], tcqa[1], 0.0]])

        return Hq

class KagomeSOBlochHamiltonian(KagomeBlochHamiltonian):
    """
    Kagome tight-binding Hamiltonian with Rashba spin-orbit coupling/TE-TM
    splitting
    """
    def __init__(self):
        """
        """
        KagomeBlochHamiltonian.__init__(self)                           #

        self.nu = 6                                                     # number of bands

    def bloch_matrix(self, q, t):
        """
        Compute the Hamiltonian matrix at wavevector q and couplings t

        Parameters
        ----------
        q : array
            wavevector
        t : array
            the couplings, 6 = 3+3 for now

        Returns
        -------
        Hq : array
            the Hamiltonian matrix
        """
        nu = self.nu//2                                                 #

        H0 = KagomeBlochHamiltonian.bloch_matrix(self, q, t[:nu])       #

        ssqa = t[nu:]*_np.sin(self.a@q)
        x0 = _cmt.exp(1j*_mt.pi/6)
        x1 = 1.0/x0

        HR = _np.array([[0.0, x0*ssqa[0],-ssqa[2]],
                        [x0*ssqa[0], 0.0, -1j*ssqa[1]],
                        [-x1*ssqa[2], -1j*ssqa[1], 0.0]], dtype=_np.complex128)

#        Z = _np.zeros((nu, nu), dtype=_np.complex128)

        Hq = _np.block([[H0, HR], [_np.conjugate(HR.T), H0]])

        return Hq

#   ?
class KagomeModel(_gen.LatticeModel):
    """
    Generic kagome tight-binding model
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.KagomeLattice(LatticeMatFile)             #

    def _ue_finalise(self, psi_t, config, data):
        """
        """
        _gen.LatticeModel._ue_finalise(self, psi_t, config, data)       # universal part

#   ?
        L = self.L                                                      # lattice
        sl = L['sl']                                                    # sublattice structure
        sl_num = L['sl_num']                                            # number of sublattices

        t_num = psi_t.shape[0]                                          #

#   ?
        if 'wf_shell' in config:                                        # wavefunction weights on lattice shells starting from (0, 0)
            (ix, ix_d, Rs) = _lsm.lattice.kagome.hex_shells_origin(L)   # lattice shells
            shell_num = ix_d.size - 1                                   # number of shells

            wf_shell_t = _np.zeros((sl_num, shell_num, t_num))          # the wavefunction on shells

            for i in range(sl_num):                                     # loop over sublattices
                for j in range(shell_num):                              # loop over shells
                    a = ix[ix_d[j]:ix_d[j+1]]                           #
                    b = a[sl[a] == i]

                    wf_shell_t[i, j, :] = _np.sum(_np.abs(psi_t[:, b])**2, axis=1)

            data['shell_r'] = Rs
            data['wf_shell_t'] = wf_shell_t

class KagomeSOModel(_gen.LatticeSOModel):
    """
    spin-orbit/TE-TM splitting model on kagome lattice
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.KagomeLattice(LatticeMatFile)             #

    def _ue_finalise(self, psi_t, config, data):
        """
        """
        _gen.LatticeSOModel._ue_finalise(self, psi_t, config, data)     # universal part

#   ?
        L = self.L                                                      # lattice
        sl = L['sl']
        sl_num = L['sl_num']

        t_num = psi_t.shape[0]                                          #

#   ?
        if 'wf_shell' in config:                                        # weight of shells and the weight chagnes according to distance and time
            (ix, ix_d, Rs) = _lsm.lattice.kagome.hex_shells_origin(L)   #
            shell_num = ix_d.size - 1                                   # number of shells

            wf_shell_t = _np.zeros((2, sl_num, shell_num, t_num))

            for i in range(sl_num):                                     # loop over sublattices
                for j in range(shell_num):                              # loop over shells
                    a = ix[ix_d[j]:ix_d[j+1]]                           #
                    b = a[sl[a] == i]

                    wf_shell_t[0, i, j, :] = _np.sum(_np.abs(psi_t[:, 2*b])**2, axis=1)
                    wf_shell_t[1, i, j, :] = _np.sum(_np.abs(psi_t[:, 2*b + 1])**2, axis=1)

            data['shell_r'] = Rs
            data['wf_shell_t'] = wf_shell_t

#   Initial wavefunctions for unitary evolution
def _wf_cls_hex_make(L, config):
    """
    Construct CLS wavefunction

    Paramters
    ---------
    L : KagomeLattice
    config : dict
    """
    assert L['tiling'] == 'hex', 'Only hexagonal tiling is supported!'

#   Parameters
    N = len(L)                                                          # number of sites
    A = L['A']                                                          # adjacency matrix
    m = config['m']                                                     # number of spin components: 1 - tight-binding, 2 - spin-orbit
    ab = config['ab']                                                   # only for spin-orbit: linear combination of up-down components

    r0 = _np.array([0.0, 0.0])
    R = L.distance_from_point(r0)
    ix = R.argsort()
    h0 = list(ix[:6])                                                   # the central hexagon indices

#   ?
    h = [h0[0]]
    nb = A.indices[A.indptr[h[0]]:A.indptr[h[0]+1]]
    i = 0
    while nb[i] not in h0:
        i += 1
    h.append(nb[i])

    h1 = [i for i in h0 if i not in h]

#   ?
    while len(h) < 6:
        i = h[-1]
        nb = A.indices[A.indptr[i]:A.indptr[i+1]]
        dh = [j for j in nb if j in h0 and j not in h]

        assert len(dh) < 2, 'Counting went wrong!'

        if len(dh) == 1:
            h.append(dh[0])

    h = _np.array(h)

#        for i in h1:
#            nb = A.indices[A.indptr[i]:A.indptr[i+1]]
#            dh = [j for j in nb if j in h0 and j not in h]

#            assert len(dh) < 2, 'Conting went wrong!'

#            if len(dh) == 1:
#                h.append(dh[0])

    phi = _np.array([2*(i%2) - 1 for i in range(6)])
    if m == 1:                                                          # tight-binding model
        psi = _np.zeros(N, dtype=_np.complex128)
        psi[h] = ab*phi
    else:                                                               # spin-orbit coupling/TE-TM splitting
        psi = _np.zeros(2*N, dtype=_np.complex128)
        psi[2*h] = ab[0]*phi
        psi[2*h + 1] = ab[1]*phi

#   ?
    data = {}
#    data['name'] = 'cls_hex'
    data['m'] = m
    data['psi'] = psi/_norm(psi)
    data['index'] = h
    data['ab'] = ab

    return data

def wf_make(L, wf_type, config):
    """
    Generate initial wavefunction for unitary evolution

    Parameters
    ----------
    L : lsm.lattice.KagomeLattice
    wf_type : str
        wavefunction type
    config : dict

    Returns
    -------
    data : dict
    """
    if wf_type == 'cls_hex':                                            # minimal CLS: single hexagon at the center of the lattice
        data = _wf_cls_hex_make(L, config)
    else:                                                               #
        data = _gen.wf_make(L, wf_type, config)

    data['time'] = 0.0                                                  #

    return data
