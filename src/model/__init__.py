#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  __init__.py
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


#   ?


#from .hamiltonian import BlochHamiltonian

from .wf import Wavefunction
from .hamiltonian import Hamiltonian

from .generic import LatticeModel, LatticeSOModel
from .kagome import KagomeModel, KagomeSOModel
from .lieb import LiebModel, LiebSOModel
from .dice import DiceModel, DiceSOModel
from .square import SquareModel
from .triangular import TriangularModel

from .kagome import KagomeBlochHamiltonian, KagomeSOBlochHamiltonian

from . import wf
from . import hamiltonian as ham

from . import dice
from . import honome
from . import kagome
from . import lieb
from . import square
from . import triangular

#   Definitions
#   Public
model_type_all = ('generic', 'kagome', 'kagome_so', 'lieb', 'lieb_so',
                  'dice', 'dice_so', 'square', 'triangular')

#   ?
def instantiate(lm_type, fn_l, fn_hm, fn_ef, fn_wf):
    """
    """
    if lm_type == 'kagome':
        lm_gen = KagomeModel
    elif lm_type == 'kagome_so':
        lm_gen = KagomeSOModel
    elif lm_type == 'lieb':
        lm_gen = LiebModel
    elif lm_type == 'lieb_so':
        lm_gen = LiebSOModel
    elif lm_type == 'dice':
        lm_gen = DiceModel
    elif lm_type == 'dice_so':
        lm_gen = DiceSOModel
    elif lm_type == 'square':
        lm_gen = SquareModel
    elif lm_type == 'triangular':
        lm_gen = TriangularModel
    else:
        raise ValueError('Unknown model type: ' + lm_type)

    M = lm_gen(fn_l, fn_hm, fn_ef, fn_wf)                               # instantiate the model

    return M

def instantiate_generic(m, fn_l, fn_hm, fn_ef, fn_wf):
    """
    """
    if m == 1:                                                          # no spins
        lm_gen = LatticeModel
    elif m == 2:                                                        #
        lm_gen = LatticeSOModel
    else:
        raise ValueError('Incorrect value of m:' + str(m))

    M = lm_gen(fn_l, fn_hm, fn_ef, fn_wf)

    return M
