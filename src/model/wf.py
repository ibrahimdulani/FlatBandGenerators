#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  wf.py
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


#   Wavefunction

import os.path as _osp

import numpy as _np
import scipy.io as _sio
from scipy.linalg import norm as _norm

#   Definitions
#   Internal
_key_all = ('psi', 'm', 'index', 'ab', 'name', 'wf_id', 'wf_serial', 'time')

#   Public


#   ?
class Wavefunction:
    """
    Wavefunction
    """
    def __init__(self, WfMatFile):
        """
        Initialise wavefunction
        """
        if isinstance(WfMatFile, str):                                  # filename
            self.load(WfMatFile)                                        #
        elif isinstance(WfMatFile, tuple):                              #
            (N, m) = WfMatFile                                          #
            self.name = ''                                              #
            self.psi = _np.zeros(N*m, dtype=_np.complex128)             #
            self.m = _np.uint8(m)                                       # number of spin components
            self.time = 0.0                                             #
            self.wf_id = _np.int32(-1)                                  #
            self.wf_serial = _np.int32(-1)                              #
        else:
            raise ValueError('Unknown initialisation data for a wavefunction')

    def __getitem__(self, key):
        """
        """
        key_val = getattr(self, key)

        return key_val

    def __setitem__(self, key, key_val):
        """
        """
        if key in _key_all:
            setattr(self, key, key_val)
        else:
            raise KeyError('Unknown key ' + key)

    def __iter__(self):
        """
        Dictionary representation of the wavefunction
        """
        key_all = (key for key in _key_all if key in dir(self))         #

        for key in key_all:                                             #
            yield (key, getattr(self, key))

    def __len__(self):
        """
        Size of the wavefunction
        """
        return self.psi.size

    def name_id(self):
        """
        Return string identifier of the spin state
        """
        if self.name in ('random', 'none'):                             # random or noname states
            s = ''
        else:
            s = self.name + '-'
        s += 'N-' + str(self.psi.size)
        if self.wf_id > -1:
            s += '-id' + str(self.wf_id)
        if self.wf_serial > -1:
            s += '-' + str(self.wf_serial)

        return s

#   Properties
    def norm(self, fmt='full'):
        """
        Copmpute the norm of the wavefunction
        """
        if fmt == 'full':                                               # full norm
            t = _norm(self.psi)
        elif self.m == 1:                                               #
            t = _np.abs(self.psi)
        elif self.m == 2:                                               #
            t = _np.sqrt(_np.abs(self.psi[::2])**2 + _np.abs(self.psi[1::2])**2)
        else:
            raise ValueError('Unknown format: ' + fmt)

        return t

#   I/O
    def load(self, WfMatFile):
        """
        Load wavefunction
        """
        data = _sio.loadmat(WfMatFile, squeeze_me=True)                 #

        for k in data:                                                  #
            if k[0] == '_':                                             # skip internal data
                continue

            setattr(self, k, data[k])                                   #

        if 'time' not in data:                                          # timestamp
            self.time = 0.0

        if 'wf_id' not in data:                                         #
            self.wf_id = _np.int32(-1)
        if 'wf_serial' not in data:                                     #
            self.wf_serial = _np.int32(-1)

    def save(self, *WFMatFile):
        """
        Save wavefunction to a .mat file
        """
#   Output filename
        if len(WFMatFile) == 0:                                         # no output filename was supplied
            fn_out = self.name_id()
        else:                                                           # output filename was supplied
            fn_out = WFMatFile[0]

        x = _osp.splitext(fn_out)                                        # extension
        if x[1] != '.mat':                                              # no extension
            fn_out += '.mat'                                            # add one

        _sio.savemat(fn_out, dict(self), do_compression=True, oned_as='row')

#   ?
def disorder_default(wfd):
    """
    """
    wfd['wf_id'] = _np.int32(-1)
    wfd['wf_serial'] = _np.int32(-1)

def ipr_get(psi, q):
    """
    """
    ipr_q = _np.sum(_np.abs(psi)**(2*q))
    ipr_q /= _np.sum(_np.abs(psi)**2)

    return ipr_q
