#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  cls.py
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


#   Compact localised state manipulation

import warnings as _wn

import numpy as _np
import scipy.linalg as _la

def sub_matrix(H, uc, i0, U):
    """
    Cut out a submatrix Hamiltonian corresponding to the U CLS problem
    Periodic b.c. implicitly assumed (via uc in computation of T1* below)

    Parameters
    ----------

    Returns
    -------

    """
    nu = uc.shape[1]                                                    # number of sites in the unit cell

    ix = _np.ravel([uc[i0 + i] for i in range(U)])                      #

#   ?
    HU = _np.zeros((ix.size, ix.size), dtype=H.dtype)
    for i in range(ix.size):
        HU[i, :] = H[i, ix]

#   Hopping matrix from the - H1
    T1 = _np.zeros((nu, nu), dtype=H.dtype)
    for i in range(nu):
        T1[i, :] = H[uc[i0 + U - 1, i], uc[i0 + U, :]]

#   h.c. of H1
    T1_hc = _np.zeros((nu, nu), dtype=H.dtype)
    for i in range(nu):
        T1_hc[i, :] = H[uc[i0, i], uc[i0 - 1, :]]

    return (HU, T1, T1_hc)

def find(H, uc, U_max, eps=1e-8, translation_invariant=False, T_all=None):
        """
        Search for possible compact localised states of the Hamiltonian
        H.
        For now only implemented for 1D n.n. hopping Hamiltonians
        Periodic b.c. implicitly assumed for uc

        Parameters
        ----------
        H : (sparse) array
            the Hamiltonian
        uc : array
            partitioning into unit cells: uc[i] sites in the cell i
        U_max : int
            maximal CLS size to search for
        eps : float, optional
            zero threshold
        translation_invariant : bool, optional
            is the Hamiltonian translation invariant?

        Returns
        -------
        E : list
            list of CLS energies
        psi : list
            list of CLS
        """
#   Parameters
        nu = uc.shape[1]                                                # number of site in unit cell
        found = False                                                   #

#   ?
        if translation_invariant:                                       # translationally invariant Hamiltonian
            i_all = (1,)

            if T_all is not None:                                       # hopping matrices provided
                T1 = T_all[1]                                           # n.n. hopping
                T1_hc = _np.conjugate(T1.T)                             # hermitian conjugate of H1

            r = None
        else:                                                           #
            psi_full = _np.zeros(H.shape[1], dtype=H.dtype)

            i_all = range(uc.shape[0])
            r = {}

#   Search for CLS
        for i in i_all:                                                 # loop over unit cells
            for U in range(1, U_max + 1):                               # loop over U class
#            HU = self.cls_ham_make(U)                                   # generate U-CLS Hamiltonian from H0 and H1
                (HU, t0, t1) = sub_matrix(H, uc, i, U)                  #

                if translation_invariant and T_all is not None:
                    assert _np.allclose(T1, t0) and _np.allclose(T1_hc, t1), 'Inconsistent hopping matrices!'
                else:
                    T1 = t0
                    T1_hc = t1

                (E, psi) = _la.eigh(HU)                                 # diagonalise

                adE = _np.abs(_np.diff(E))                              #
                mp = _np.where(adE < eps)[0]                            # doubley degenerate eigenvalues

                if mp.size > 0:                                         # there are degenerate eigenvalues
                    _wn.warn('Degenerate eigenvalue(s) detected: results might be inaccurate!')

#                for i in mp:                                            #
#                    psi = _detangle(psi, i, H1)

#   Psi_1
                u1 = T1@psi[:nu, :]                                     # u1_ak = H1_ab psi_bk
                u1n = _np.sqrt(_np.sum(_np.abs(u1), axis=0)/nu)         # u1n_k = sum_a u1_ak

#   Psi_U
                uu = T1_hc@psi[-nu:, :]
                uun = _np.sqrt(_np.sum(_np.abs(uu), axis=0)/nu)

#   CLS test
                z = _np.array([i for i in range(HU.shape[0]) if u1n[i] < eps and uun[i] < eps])
#            z = _np.array([i for i in ix1 if i in ixu])                 # rows that are zero for both left and right
                if z.size > 0:                                          # there are zero rows, that are left and right at the same time
                    found = True                                        #

                    break
            if not translation_invariant:                               #
                t = []
                for j in z:                                             #
                    psi_full[:] = 0.0
                    k = i*nu
                    psi_full[k:k+U*nu] = psi[:, j]
                    t.append((E[i], psi_full.copy()))
                r[i] = t

#   Finalise the output
        if translation_invariant and found:                             # found a CLS
            r = [(E[i], psi[:, i]) for i in z]

        return r
