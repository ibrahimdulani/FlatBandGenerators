#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  lieb.py
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


#   Lieb lattice models

import numpy as _np
from scipy.linalg import norm as _norm

import lsm as _lsm

from . import generic as _gen
from . import wf as _wf

#   Definitions
#   Internal


#   Public
wf_type_all = ('ll_cls',)                                               #

#   ?
class LiebModel(_gen.LatticeModel):
    """
    Lieb lattice model
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.LiebLattice(LatticeMatFile)               #

#    def _ue_finalise(self, psi_t, config, data):
#        """
#        """
#        _LatticeModel._ue_finalise(psi_t, config, data)                 # universal part

#   ?
#        L = self.L                                                      # lattice
#        sl = L['sl']                                                    # sublattice structure
#        sl_num = L['sl_num']                                            # number of sublattices

#        t_num = psi_t.shape[0]                                          # number of points in time

class LiebSOModel(_gen.LatticeSOModel):
    """
    Lieb lattice model with SO coupling
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.LiebLattice(LatticeMatFile)               #

#    def _ue_finalise(self, psi_t, config, data):
#        """
#        """
#        _LatticeModel._ue_finalise(psi_t, config, data)                 # universal part

#   ?
#        L = self.L                                                      # lattice
#        sl = L['sl']                                                    # sublattice structure
#        sl_num = L['sl_num']                                            # number of sublattices

#        t_num = psi_t.shape[0]                                          # number of points in time

#   ?

#
#   Initial wavefunctions for unitary evolution
def _wf_cls_square_make(L, config):
    """
    Construct CLS wavefunction on the Lieb lattice

    Parameters
    ----------
    L : LiebLattice
    config : dict
    """
    N = len(L)                                                          # number of sites
    a = L.spacing()                                                     # lattice spacing
    sq_size = 4                                                         #
    eps = 1e-3                                                          # zero threshold

#   Sanity checks
#    assert _np.all(L['uc_t']%2 == 1), 'Number of unit cell translations has to be odd!'
    assert abs(a - 1.0) < eps, 'Lattice spacing is not 1.0!'

#   ?
    r = L['r']                                                          # lattice sites
    A = L['A']                                                          # adjacency matrix
    sl = L['sl']                                                        # sublattice structure
    ls = L['ls']                                                        #
    sq_rc = L['fu_rc']                                                  #
    sq_lp = L['fu_lp']                                                  #

    m = config['m']                                                     # number of spin components: 1 - tight-binding, 2 - spin-orbit
    ab = config['ab']                                                   # only for spin-orbit: linear combination of up-down components
    if 'r_cls' in config:                                               # CLS position was specified
        r_cls = config['r_cls']
    else:                                                               # use the center of the lattice
        r_cls = ls/2

    ix = _np.where(_np.abs(sq_rc[:, 0] - r_cls[0]) < eps)[0]
    ix1 = _np.where(_np.abs(sq_rc[ix, 1] - r_cls[1]) < eps)[0]
    k0 = int(ix[ix1])                                                   # square index

    r0 = sq_rc[k0]                                                      # center of the square
    sq0 = sq_lp.indices[sq_lp.indptr[k0]:sq_lp.indptr[k0+1]]            #
    sq1 = _np.array([i for i in sq0 if _norm(r[i] - r0) < a*(1 + eps)])
    sq2 = _np.array([i for i in sq0 if i not in sq1])

    assert _np.all(sl[sq1] == 1), 'Incorrect CLS sites!'

    p = _np.zeros(sq_size, dtype=_np.int32)
    p[0] = sq1[_np.argmin(r[sq1, 1])]
    p[1] = sq1[_np.argmin(r[sq1, 0])]
    p[2] = sq1[_np.argmax(r[sq1, 1])]
    p[3] = sq1[_np.argmax(r[sq1, 0])]

    phi = _np.array([2*(i%2) - 1 for i in range(sq_size)])
    if m == 1:                                                          # tight-binding model
        psi = _np.zeros(N, dtype=_np.complex128)
        psi[p] = ab*phi
    else:                                                               # spin-orbit coupling/TE-TM splitting
        psi = _np.zeros(2*N, dtype=_np.complex128)
        psi[2*p] = ab[0]*phi                                            # spin down
        psi[2*p + 1] = ab[1]*phi                                        # spin up

#   ?
    data = {}
#    data['name'] = 'cls_square'
    data['psi'] = psi/_norm(psi)
    data['index'] = p
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
    if wf_type == 'cls_square':                                         # minimal CLS: single square at the center of the lattice
        data = _wf_cls_square_make(L, config)
    else:                                                               #
        data = _gen.wf_make(L, wf_type, config)

    data['time'] = 0.0

    return data
