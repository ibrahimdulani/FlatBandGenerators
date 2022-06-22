#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  honome.py
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


#   Tight binding models on honome lattice

import numpy as _np
import scipy.sparse as _ssp

import lsm as _lsm

from .generic import LatticeModel as _LatticeModel

#   Definitions
#   Internal


#   Public


#   ?
class HonomeModel(_LatticeModel):
    """
    Tight-binding model on the honome lattice
    """
    def _init_lattice(self, LatticeMatFile):
        """
        Initialise the lattice
        """
        self.L = _lsm.lattice.HonomeLattice(LatticeMatFile)             # load the honome lattice

#   Tight-binding Hamiltonian
def hm_make(L, e0, t0, eps=1e-6):
    """
    Tight binding Hamiltonian on honome lattice

    Parameters
    ----------
    L : lsm.lattice.HonomeLattice
    e0 : array
        onsite energies, per sublattice, 0 and 1
    t0 : array
        the hoppings
        distance         sl         coupling index  distance index
            5.773503e-01    012-3       0               0
            1.0             012-012     1               1
            1.154701e+00    3-3         2               2
            1.527525e+00    012-3       3               3
            1.732051e+00    012-012     4               4
            2.0             012-012     5               5
                            3-3         6
            2.081666e+00    012-3       7               6
            2.309401e+00    3-3         8               7
            2.516611e+00    012-3       9               8
    eps : float, optional
        zero threshold

    Returns
    -------
    M : sparse array
        the Hamiltonian matrix
    """
#   Definitions
    N = len(L)                                                          # number of sites
    r = L['r']                                                          # lattice sites
    sl = L['sl']                                                        # sublattice structure

    R_tb = _np.array([5.773503e-01, 1.0, 1.154701e+00, 1.527525e+00,
                      1.732051e+00, 2.0, 2.081666e+00, 2.309401e+00,
                      2.516611e+00])                                    #

    M = _ssp.lil_matrix((N, N), dtype=_np.float64)                      # The Hamiltonian matrix

#   Set the onsite energies and the bonds
    b = 0                                                               # coupling index
    for i in range(N):                                                  # loop over all sites
        M[i, i] = e0[sl[i] == 3]                                        # the onsite energies

#   The bonds
        R = L.distance_from_point(r[i])                                 # distance to all the other sites from the current one
        p = R.argsort()                                                 # sorting permutation
#        Rp = R[p]
        ix = _np.where(R < R_tb[-1]*1.03)[0]                            # hopping neighbours

        for j in ix:                                                    # loop over the neighbours
            if abs(R[j]) < eps:
                continue

            sl_eq = (sl[i] < 3 and sl[j] < 3)
            sl_eq3 = (sl[i] == 3 and sl[j] == 3)
            sl_diff = ((sl[i] == 3 and sl[j] < 3) or (sl[i] < 3 and sl[j] == 3))

            a = int(_np.where(_np.abs(R_tb - R[j]) < eps)[0])           #
            if a == 0:                                                  # 5.773503e-01
                assert sl_diff, 'Incorrect sublattice structure: 0'

                b = 0
            elif a == 1:                                                # 1.0
                assert sl_eq, 'Incorrect sublattice structure: 1'

                b = 1
            elif a == 2:                                                # 1.154701e+00
                assert sl_eq3, 'Incorrect sublattice structure: 2'

                b = 2
            elif a == 3:                                                # 1.527525e+00
                assert sl_diff, 'Incorrect sublattice structure: 3'

                b = 3
            elif a == 4:                                                # 1.732051e+00
                assert sl_eq, 'Incorrect sublattice structure: 4'

                b = 4
            elif a == 5:                                                # 2.0
                assert sl_eq or sl_eq3, 'Incorrect sublattice structure 5'

                if sl_eq:
                    b = 5
                else:
                    b = 6
            elif a == 6:                                                # 2.081666e+00
                assert sl_diff, 'Incorrect sublattice structure: 6'

                b = 7
            elif a == 7:                                                # 2.309401e+00
                assert sl_eq3, 'Incorrect sublattice structure: 7'

                b = 8
            elif a == 8:                                                # 2.516611e+00
                assert sl_diff, 'Incorrect sublattice structure: 8'

                b = 9
            else:
                raise ValueError('Something went wrong!')

            M[i, j] = t0[b]
            M[j, i] = t0[b]

    return M.tocsr()
