#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  dice.py
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


#   Dice model with or without spin-orbit coupling

import cmath as _cmt

import numpy as _np
from scipy.linalg import norm as _norm

import lsm as _lsm

from . import generic as _gen

#   Definitions
#   Internal


#   Public
wf_type_all = ('cls_hex',)                                              #

#   ?
class DiceModel(_gen.LatticeModel):
    """
    Generic tight-binding model on the dice/T3 lattice
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.DiceLattice(LatticeMatFile)


class DiceSOModel(_gen.LatticeSOModel):
    """
    Generic tight-binding model on the dice/T3 lattice with spin-orbit
    coupling
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.DiceLattice(LatticeMatFile)

#   Generate wavefunctions
def _wf_cls_hex_make(L, config, eps=1e-2):
    """
    """
    N = len(L)                                                          # number of sites
    r = L['r']                                                          # lattice sites

    m = config['m']                                                     # number of spin components: 1 - tight-binding, 2 - spin-orbit
    ab = config['ab']                                                   # only for spin-orbit: linear combination of up-down components
    l_so = config['lambda']

    r2 = _np.array([[-0.8660, -2.5], [-1.7321, -2.0],
                    [-1.7321, -1.0], [-2.5981, -0.5],
                    [0.8660 , -2.5], [   0   , -2.0],
                    [0      , -1.0], [-0.8660, -0.5],
                    [1.7321 , -2.0], [1.7321 , -1.0],
                    [2.5981 , -0.5], [0.8660 , -0.5],
                    [-0.8660, 2.5], [-1.7321, 2.0],
                    [-1.7321, 1.0], [-2.5981, 0.5],
                    [0.8660 , 2.5], [0      , 2.0],
                    [0      , 1.0], [-0.8660, 0.5],
                    [ 1.7321, 2.0], [ 1.7321, 1.0],
                    [2.5981 , 0.5], [0.8660 , 0.5],
                    [-0.8660, -1.5], [-1.7321, 0.0],
                    [0.8660 , -1.5], [0.0 , 0.0],
                    [1.7321 , 0.0], [-0.8660, 1.5],
                    [ 0.8660, 1.5]])                                    # the L=2 hexagonal dice lattice

#   Sanity checks
    assert L['tiling'] == 'hex', 'Only hexagonal tilings are allowed!'
    assert m == 2, 'Only spin-orbit coupled configurations are allowed!'

#   Find the embedding of r2(CLS) into the larger lattice
    R = _np.zeros(N)
    p = _np.zeros(r2.shape[0], dtype=_np.int32)                         #

    for i in range(r2.shape[0]):
        R[:] = _np.array([_norm(r[j] - r2[i]) for j in range(N)])       #
        p[i] = _np.int(_np.where(R < eps)[0])                           #

#   Generate wavefunction
    if m == 1:                                                          # tight-binding model
        raise NotImplementedError('Tight-binding CLS has not been implemented for the dice lattice!')
#        psi2 = _np.array([[], []], dtype=_np.complex128)                # the CLS

        psi = _np.zeros(N, dtype=_np.complex128)
        psi[p] = ab*psi2
    else:
                                                                        # spin-orbit coupling/TE-TM splitting
        psi2 = _np.array([[_cmt.exp(5j*_cmt.pi/6),
                           _cmt.exp(-1j*_cmt.pi/6),
                           -_cmt.sqrt(3),
                           _cmt.exp(1j*_cmt.pi/6),
                           1j,                                          #5
                           _cmt.sqrt(3)*_cmt.exp(-1j*_cmt.pi/3),
                           _cmt.sqrt(3)*_cmt.exp(2j*_cmt.pi/3),
                           _cmt.sqrt(3),
                           -1j,
                           _cmt.sqrt(3)*_cmt.exp(1j*_cmt.pi/3),         #10
                           _cmt.exp(-5j*_cmt.pi/6),
                           _cmt.sqrt(3)*_cmt.exp(-2j*_cmt.pi/3),
                           1j,
                           -1j,
                           _cmt.sqrt(3)*_cmt.exp(1j*_cmt.pi/3),         #15
                           _cmt.exp(-5j*_cmt.pi/6),
                           _cmt.exp(5j*_cmt.pi/6),
                           _cmt.sqrt(3)*_cmt.exp(-1j*_cmt.pi/3),
                           _cmt.sqrt(3)*_cmt.exp(2j*_cmt.pi/3),
                           _cmt.sqrt(3)*_cmt.exp(-2j*_cmt.pi/3),        #20
                           _cmt.exp(-1j*_cmt.pi/6),
                           -_cmt.sqrt(3),
                           _cmt.exp(1j*_cmt.pi/6),
                           _cmt.sqrt(3),
                           0,0,0,0,0,0,0],
                           [_cmt.exp(-2j*_cmt.pi/3)*l_so,
                            _cmt.exp(-1j*_cmt.pi/3)*l_so,
                            0,
                            _cmt.exp(-2j*_cmt.pi/3)*l_so,
                            _cmt.exp(-1j*_cmt.pi/3)*l_so,              #5
                            0,
                            _cmt.sqrt(3)*1j*l_so,
                            _cmt.sqrt(3)*1j*l_so,
                            _cmt.exp(-2j*_cmt.pi/3)*l_so,
                            0,                                          #10
                            _cmt.exp(-1j*_cmt.pi/3)*l_so,
                            _cmt.sqrt(3)*1j*l_so,
                            _cmt.exp(-1j*_cmt.pi/3)*l_so,
                            _cmt.exp(-2j*_cmt.pi/3)*l_so,
                            0,                                          #15
                            _cmt.exp(-1j*_cmt.pi/3)*l_so,
                            _cmt.exp(-2j*_cmt.pi/3)*l_so,
                            0,
                            _cmt.sqrt(3)*1j*l_so,
                            _cmt.sqrt(3)*1j*l_so,                       #20
                            _cmt.exp(-1j*_cmt.pi/3)*l_so,
                            0,
                            _cmt.exp(-2j*_cmt.pi/3)*l_so,
                            _cmt.sqrt(3)*1j*l_so,
                            0, 0, 0, 0, 0, 0, 0]], dtype=_np.complex128)# the CLS

        psi = _np.zeros(2*N, dtype=_np.complex128)
        psi[2*p] = ab[0]*psi2[0, :]                                     # spin down component
        psi[2*p + 1] = ab[1]*psi2[1, :]                                 # spin up component

#   Finalise
#    data['name'] = 'cls_hex'
    data = {}
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
    if wf_type == 'cls_hex':                                            #
        data = _wf_cls_hex_make(L, config)
    else:                                                               #
        data = _gen.wf_make(L, wf_type, config)

    data['time'] = 0.0                                                  #

    return data
