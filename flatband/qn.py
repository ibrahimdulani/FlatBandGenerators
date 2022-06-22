#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  qn.py
#
#  Copyright 2016-2017 alexei andreanov <alexei.andreanov@gmail.com>
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


#   Flatband construction and testing

import math as _mt
import cmath as _cmt
import warnings as _wn
from os.path import splitext as _splitext

import numpy as _np
import scipy.io as _sio
import scipy.linalg as _la
import scipy.optimize as _sop

#   ?
def _canonicalise(H0, H1, eps=1e-4):
    """
    Transform input matrices to canonical form
    """
    (E, psi) = _la.eigh(H0)                                             # diagonalise the intracell hopping matrix
    E -= E[0]                                                           # shift the lowest eigenvalue to zero
    ix = _np.where(E > eps)[0]                                          # indices of zero eigenvalues
    E1 = E[ix[0]]                                                       #
    if ix.size > 0:                                                     # non-degenerate
        E /= E1

#    G0 = psi.T@H0@psi
    G0 = _np.diag(E)                                                    # H0 in canonical form
    G1 = psi.T@H1@psi/E1                                                # H1 in canonical form

    return (G0, G1)

def _detangle(psi, i, H1, eps=1e-10):
    """
    Find a rotation that satisfied the "no-leak" condition
    """
#   Parameters
    nu = H1.shape[0]                                                    # number of bands
    H1_hc = _np.conjugate(H1.T)                                         # Hermitian conjugate

#   ?
    x0 = H1@psi[:nu, i]
    x1 = H1@psi[:nu, i+1]
    y0 = H1_hc@psi[-nu:, i]
    y1 = H1_hc@psi[-nu:, i+1]

#   ?
    M0 = _np.array([[x0.conjugate()@x0, x0.conjugate()@x1], [x1.conjugate()@x0, x1.conjugate()@x1]])
    M1 = _np.array([[y0.conjugate()@y0, y0.conjugate()@y1], [y1.conjugate()@y0, y1.conjugate()@y1]])

#   ?
    M0_use = False
    M1_use = False
    if _np.linalg.matrix_rank(M0) == 1 and not _np.allclose(M0, 1):
        (E0, u0) = _la.eigh(M0)
        ix0 = _np.where(_np.abs(E0) > eps)[0]
        M0_use = True
    if _np.linalg.matrix_rank(M1) == 1:
        (E1, u1) = _la.eigh(M1)
        ix1 = _np.where(_np.abs(E1) > eps)[0]
        M1_use = True

#   ?
    phi = psi.copy()
    if M0_use and M1_use:                                               #
        t = u1[:, ix1[0]]/u0[:, ix0[0]]
        if _np.abs(t[0] - t[1]) < eps:
            phi[:, i] = u0[0, ix0[0]]*psi[:, i] + u0[1, ix0[0]]*psi[:, 1]
            phi[:, i] /= _la.norm(phi[:, i])
    elif (not M0_use) and (not M1_use):                                 #
        pass
    elif M0_use:
        phi[:, i] = u0[0, ix0[0]]*psi[:, i] + u0[1, ix0[0]]*psi[:, 1]
        phi[:, i] /= _la.norm(phi[:, i])
    elif M1_use:
        phi[:, i] = u1[0, ix1[0]]*psi[:, i] + u0[1, ix1[0]]*psi[:, 1]
        phi[:, i] /= _la.norm(phi[:, i])
    else:
        raise ValueError('This branch should never be executed!')

    return phi

#   ?
class QuantumNetwork():
    """
    Generic d=1 quantum network
    """
    def __init__(self, QN_data, eps=1e-8, canonical=True):
        """
        Network constructor

        Parameters
        ----------
        H_data : tuple
            H0, H1, ... : array
                container of all the hopping matrices
                should have the size (1 + mc)xnuxnu
        eps : float, optional
            zero threshold
        canonical : bool, optional
            convert hopping matrices to canonical form
        """
        self.canonical = canonical                                      #

        if isinstance(QN_data, str):                                    # str == filename
            (fn_base, fn_ext) = _splitext(QN_data)                      #

            if fn_ext == '.hp':                                         # text data
                H_data = _np.loadtxt(QN_data, dtype=_np.complex128)
            elif fn_ext in ('.npy', '.npz'):                            # NumPy binary format
                H_data = _np.load(QN_data, allow_pickle=False)
            elif fn_ext == '.mat':                                      # MATLAB files
                data = sio.loadmat(QN_data, squeeze_me=True)
                H_data = data['H_data']
            else:
                raise ValueError('Unknown filetype' + fn_ext)

            H0 = H_data[0]                                              # intracell hopping
            if _np.allclose(H0, _np.diag(_np.diag(H0))) and _np.allclose(H0.diagonal(), _np.sort(H0.diagonal())):
                self.canonical = True
            else:                                                       #
                self.canonical = False                                  # never set this flag if loading matrices from files
        else:                                                           # array, tuple, ...
            H_data = QN_data
        self.H_all = _np.array(H_data, dtype=_np.complex128)            # hopping matrices

#   Generic part
        H0 = self.H_all[0]                                              # intracell hopping
        self.nu = H0.shape[0]                                           # number of sites in the cell
        self.mc = self.H_all.shape[0] - 1                               # hopping range
        self.eps = eps                                                  #

#   Sanity checks
        assert self.mc == 1, 'Only n.n. hopping has been implemented!'
        assert _np.all(_np.abs(H0 - _np.conjugate(H0.T)) < self.eps), 'Intracell hopping is not Hermitian!'

#   Canonicalisation of the hopping matrices
        if canonical:                                                   #
            (H0, H1) = _canonicalise(self.H_all[0], self.H_all[1])

    def __getitem__(self, key):
        """
        Dict like access to internal data
        """
        if key in ('nu', 'mc', 'canonical'):                            # number of bands, hopping range,
            key_val = getattr(self, key)
        elif key == 'H0':                                               # intracell hopping
            key_val = self.H_all[0]
        elif key == 'H1':                                               # n.n. hopping to the left
            key_val = self.H_all[1]
        else:
            raise KeyError('Unknown key ' + key)

        return key_val

#   I/O
#    def load

#   ?
    def cls_ham_make(self, U):
        """
        Generate Hamiltonian on U unit cells, assuming wavefunctions are
        zero outside the cells

        Parameters
        ----------
        U : int
            number of the unit cells

        Returns
        -------
        HU : array
            the Hamiltonian matrix
        """
#   Parameters
        H0 = self.H_all[0]                                              # intracell hopping
        H1 = self.H_all[1]                                              # n.n. cell hopping

        HU = [[_np.zeros_like(H0) for j in range(U)] for i in range(U)] # empty Hamiltonian matrix

#   Construct the block Hamiltonian
        for i in range(U):                                              # loop over the unit cells
            if i > 0:                                                   # exclude the first row
                HU[i][i-1] = _np.conjugate(H1.T)                        # H1^dagger
            HU[i][i] = H0                                               # diagonal element
            if i < U-1:                                                 # exclude the last row
                HU[i][i+1] = H1                                         #

        HU = _np.bmat(HU)                                               # construct array from blocks

#   Sanity check
        assert _np.all(_np.abs(HU - _np.conjugate(HU.T)) < self.eps), 'HU matrix is not Hermitian!'

        return HU

    def bands(self, k_num=50):
        """
        Compute the bands

        Parameters
        ----------
        k_num : int
            (-pi, pi) discretisation density

        Returns
        -------
        k_all : array
            the first Brillouin zone
        Ek_all : array
            the bands
        """
#   Parameters
        H0 = self.H_all[0]                                              # intracell hopping
        H1 = self.H_all[1]                                              # n.n. intercell hopping

        Mk = _np.zeros((self.nu, self.nu), dtype=_np.complex128)
        k_all = _np.linspace(-_mt.pi, _mt.pi, k_num, endpoint=False)    # first Brillouin zone
        Ek_all = _np.zeros((self.nu, k_num))                            # the bands

#   Compute the bands
        i = 0
        for k in k_all:                                                 # loop over the momenta
            Mk[:, :] = _cmt.exp(1j*k)*_np.conjugate(H1.T) + H0 + _cmt.exp(-1j*k)*H1
            Ek_all[:, i] = _la.eigvalsh(Mk)                             # compute the spectrum

            i += 1

        return (k_all, Ek_all)

    def fb_find(self, eps=1e-8, k_num=50):
        """
        Find flatbands

        Parameters
        ----------
        eps : float
            zero threshold
        k_num : int
            (-pi, pi) discretisation density

        Returns
        -------
        ix : array
            indices of the flatbands
        """
#   Sanity checks
        assert self.H_all.shape[0] == 2, 'Only n.n. hopping has been implemented!'

#   Detect flatbands
        (k_all, Ek_all) = self.bands(k_num=k_num)                       # compute the bands
        Ek_std = Ek_all.std(axis=1)                                     # the bandwidth
        ix = _np.where(Ek_std < eps)[0]                                 # flatband indices

        return (_np.array(ix), Ek_std)

    def has_fb(self, eps=1e-8, k_num=50):
        """
        Check whether the network has flatband(s)

        Parameters
        ----------
        eps : float
            zero threshold
        k_num : int
            (-pi, pi) discretisation density

        Returns
        -------
        ix : array
            indices of the flatbands
        """
        (ix, E_std) = self.fb_find(eps=eps, k_num=k_num)

        return (ix.size > 0)

    def cls_find(self, U_max, eps=1e-8):
        """
        Search for possible compact localised states

        Parameters
        ----------
        U_max : int
            maximal CLS size to search for
        eps : float, optional
            zero threshold

        Returns
        -------
        E : list
            list of CLS energies
        Psi : list
            list of CLS
        """
#   Parameters
        nu = self.nu                                                    # number of site in unit cell
        found = False                                                   #

        H0 = self.H_all[0]                                              # intracell hopping
        H1 = self.H_all[1]                                              # n.n. hopping
        H1_hc = _np.conjugate(H1.T)                                     # hermitian conjugate of H1
#        Z = _np.zeros_like(H0)                                          # zero matrix

#   Sanity check
        assert self.mc == 1, 'Only n.n. hopping has been implemented so far!'

#   ?
        for U in range(1, U_max):                                       # loop over U class
            HU = self.cls_ham_make(U)                                   # generate U-CLS Hamiltonian from H0 and H1
            (E, psi) = _la.eigh(HU)                                     # diagonalise

            adE = _np.abs(_np.diff(E))
            mp = _np.where(adE < eps)[0]                                # doubley degenerate eigenvalues

            if mp.size > 0:                                             # there are degenerate eigenvalues
                _wn.warn('Degenerate eigenvalue(s) detected: results might be inaccurate!')

#                for i in mp:                                            #
#                    psi = _detangle(psi, i, H1)

#   Psi_1
            u1 = H1@psi[:nu, :]                                         # u1_ak = H1_ab psi_bk
            u1n = _np.sqrt(_np.sum(_np.abs(u1), axis=0)/nu)             # u1n_k = sum_a u1_ak

#   Psi_U
            uu = H1_hc@psi[-nu:, :]
            uun = _np.sqrt(_np.sum(_np.abs(uu), axis=0)/nu)

#   CLS test
            z = _np.array([i for i in range(HU.shape[0]) if u1n[i] < eps and uun[i] < eps])
#            z = _np.array([i for i in ix1 if i in ixu])                 # rows that are zero for both left and right
            if z.size > 0:                                              # there are zero rows, that are left and right at the same time
                found = True                                            #

                break

#   Finalise the output
        if found:                                                       # found a CLS
            r = [(E[i], psi[:, i]) for i in z]
        else:                                                           # no CLS was found
            r = None

        return r
