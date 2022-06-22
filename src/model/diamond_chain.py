#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  diamond_chain.py
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


#   Hopping model on diamond chain

import lsm as _lsm

from . import generic as _gen

#   Definitions
#   Internal


#   Public


#   ?
class DiamonChainModel(_gen.LatticeModel):
    """
    Hopping model on diamond chain
    """
    def _init_lattice(self, LatticeMatFile):
        """
        ?
        """
        self.L = _lsm.lattice.DiamondChain(LatticeMatFile)              #

#   ?
