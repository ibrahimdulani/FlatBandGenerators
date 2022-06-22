#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  generic.py
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


#   Generic hopping model on a lattice

import math as _mt

import numpy as _np
import scipy.linalg as _la
import scipy.sparse as _ssp

import lsm as _lsm

from . import wf as _wf
from . import hamiltonian as _ham

#   Definitions
#   Internal
_cls_eps = 1e-8                                                         #

#   Public
spin_label_all = (r'$\downarrow$', r'$\uparrow$')                       # 0 - spin down, 1 - spin up
spin_name_all = ('down', 'up')                                          #
cls_solver_all = ('iter', 'eig', 'ue', 'iter2')                         #
ue_method_all = ('ED', 'Pade', 'emm')                                   # unitary evolution methods
wf_type_all = ('circle', 'circle_graph')                                #

#   ?
class LatticeModel:
    """
    Generic tight-binding lattice model
    """
    def _init_lattice(self, LatticeMatFile):
        """
        """
        self.L = _lsm.lattice.Lattice(LatticeMatFile)

    def _init_local(self):
        """
        """
        if 'm' not in dir(self.wf):                                     #
            self.wf.m = _np.uint8(1)

    def _sanity_local(self):
        """
        """
        assert len(self.L) == len(self.H), 'Lattice and Hamiltonian sizes mismatch!'

    def __init__(self, LatticeMatFile, HMMatFile, EFMatFile, WfMatFile):
        """
        """
        self._init_lattice(LatticeMatFile)                              # load the lattice
        self.H = _ham.Hamiltonian(HMMatFile, EFMatFile)                 # load the Hamiltonian
        self.wf = _wf.Wavefunction(WfMatFile)                           # load the wavefunction

        self._init_local()                                              # local initialisations

#   Sanity checks
        M = self.H['M']                                                 # the Hamiltonian matrix
        psi = self.wf['psi']                                            # the wavefunction

        assert M.shape[0] == M.shape[1], 'Hamiltonian matrix is not rectangular!'
        assert M.shape[0] == psi.size, 'Hamiltonian matrix and initial wavefunctions mismatch!'

        self._sanity_local()                                            # model specific checks

    def __len__(self):
        """
        """
        return len(self.L)

    def __getitem__(self, key):
        """
        """
        if key in ('L', 'H', 'wf'):                                     # model keys
            key_val = getattr(self, key)
        elif key in _lsm.lattice.generic._key_all:                      # lattice keys
            key_val = self.L[key]
        elif key in _ham._hm_key_all:                                   # Hamiltonian keys
            key_val = self.H[key]
        else:                                                           # wavefunction keys
            key_val = self.wf[key]

        return key_val

#   Properties
    def is_spinfull(self):
        """
        """
        return (len(self.L) < len(self.wf))

#   ?
    def name_id(self):
        """
        Return string identifier of the model
        """
        s = self.wf.name_id() + '-' + self.L.name_id() + '-' + self.H.name_id()#

        return s

#   ?
    def load(self, fmt, fn):
        """
        """
        if fmt in ('ef', 'hm'):                                         # external fields or Hamiltonian matrix
            self.H.load(fmt, fn)
        elif fmt == 'wf':                                               # wavefunction
            self.wf.load(fmt, fn)
        else:
            raise NotImplementedError('Loading {} has not beem implemented yet!'.format(fmt))

#   ?
    def psi_index(self, ix):
        """
        Convert lattice indices into the wavefunction indices

        Parameters
        ----------
        ix : array
            the lattice indices

        Returns
        -------
        ix_psi : array
            the wavefunction indices
        """
        m = self.wf['m']                                                # number of spin components

        ix_psi = psi_lattice_index(ix, m)                               #

        return ix_psi

#    def lattice_index(self, ix):
#        """
#        Convert
#        """

#   Unitary evolution
    def _ue_finalise(self, psi_t, config, data):
        """
        """
        if 'wf_full_keep' in config:                                    # keep the full unitary evolution
            data['psi_t'] = psi_t
        else:                                                           # only keep the last step of the unitary evolution
            data['psi_max'] = psi_t[-1]

#   ?
        L = self.L                                                      # lattice
        sl = L['sl']                                                    # sublattice structure
        sl_num = L['sl_num']                                            # number of sublattices

        t_num = psi_t.shape[0]                                          # number of time steps

#   ?
        if 'wf_b' in config:                                            # wavefunction weight on the boundary as a function of time
#            b_ix = _lsm.lattice.generic.boundary_index(L)               # sites on the boundary of the lattice
            b_ix = L.boundary()                                         # indices of the sites on the boundary of the lattice

            data['wf_b_t'] = _np.sum(_np.abs(psi_t[:, b_ix])**2, axis=1)

        if 'wf_index' in config:                                        # wavefunction weight on given set of sites - wf_index
            ix = config['wf_index']                                     # indices of the sites in the set

            data['wf_index_t'] = _np.transpose(_np.abs(psi_t[:, ix])**2)#

        if 'wf_shell_fu' in config or 'wf_shell_set' in config:         # compute evolution of weights on shells around given frustrated unit or set
            if 'wf_shell_fu' in config:                                 #
                assert L.is_frustrated(), 'Non-frustrated lattice!'

                r0 = config['wf_shell_fu']                              #
                if isinstance(r0, list):                                #
                    fu_rc = L['fu_rc']                                  #
                    R = _np.sqrt(_np.sum((fu_rc - r0)**2, axis=1))
                    fu = R.argmin()
                elif isinstance(r0, int):
                    fu = r0
                else:
                    raise ValueError('Unsuppoerted input!')

                fu_lp = L['fu_lp']
                s0_ix = fu_lp.indices[fu_lp.indptr[fu]:fu_lp.indptr[fu+1]]

                key_r = 'shell_fu'
                key_t = 'wf_shell_fu_t'
            else:                                                       #
                s0_ix = config['wf_shell_set']                          #

                key_r = 'shell_set'                                     #
                key_t = 'wf_shell_set_t'                                #

#   ?
            sh_ix = _lsm.lattice.generic.shell_index(L, s0_ix)          # compute the shells
            shell_num = 1 + sh_ix.max()                                 # number of shells

            wf_shell_t = _np.zeros((sl_num, shell_num, t_num))

            for i in range(shell_num):                                  # loop over the shells
                a = _np.where(sh_ix == i)[0]                            # the current shell

                for j in range(sl_num):                                 # loop over sublattices
                    b = a[sl[a] == j]                                   # part of the shell belonging to sublattice i

                    wf_shell_t[j, i, :] = _np.sum(_np.abs(psi_t[:, b])**2, axis=1)

            data[key_r] = sh_ix
            data[ket_t] = wf_shell_t

        if 'ipr' in config:                                             # inverse participation ratio (IPR)
            q_all = config['ipr']                                       # the IPR exponents
            q_num = q_all.size                                          # number of the exponents

            data['q_all'] = q_all
            data['ipr_t'] = _np.array([[_wf.ipr_get(psi_t[i, :], q) for i in range(t_num)] for q in q_all])

    def unitary_evolution(self, config, method='ED', sp=None):
        """
        unitary evolution of initial wavefunction psi0 with Hamiltonian
        matrix M for a discrete set of times t_all

        Parameters
        ----------
        config : dict
        method : str, optional
            how the evolution operator U is computed
                'ED'   - exact diagnalisation of the Hamiltonian matrix
                'Pade' - using Pade approximants for the evolution operator
                         Quite slow
                'emm'  - using scale and square algorithm, fast
        sp : dict or None, optional
            precomputed spectrum of the Hamiltonian matrix, only used by
            the ED method

        Returns
        -------
        data : dict
        """
#   Initialisation
        M = self.H['M']                                                 # the Hamiltonian matrix
        psi0 = self.wf['psi']                                           # the wavefunction

        N = len(self)                                                   # number of sites

        t_min = self.wf['time']                                         # initial time
        t_min += config['time'][0]                                      # additional shift
        config['t_min'] = t_min
        t_max = t_min + config['time'][1]                               # final fime
        config['t_max'] = t_max
        t_num = _np.int(config['time'][2])                              # number of time steps
        config['t_num'] = t_num

        t_all = _np.linspace(t_min, t_max, t_num)                       # times
        psi_t = _np.zeros((t_num, psi0.size), dtype=_np.complex128)     #

#   Unitary evolution
        if method == 'ED':                                              # exact diagonalisation
            if sp is None:                                              # no precomputed spectrum provided
                assert M.shape[0] < 2048, 'Hamiltonian matrix is too large for the current naive unitary evolution!'

                if _ssp.issparse(M):                                    # sparse matrix
                    M_dense = M.todense()
                else:                                                   # dense matrix
                    M_dense = M

                (E, phi) = _la.eigh(M_dense)                            # diagonalise the Hamiltonian matrix: E, |phi>
            else:                                                       # use precomputed spectrum
                E = sp['E']
                phi = sp['psi']

                assert E.size == M.shape[0], 'Wrong number of eigenvalues!'
                assert phi.shape[1] == M.shape[0], 'Wrong number of eigenvectors!'

            w = psi0@phi.conjugate()                                    # w_a = <phi_a|psi0>
            z = _np.zeros_like(E, dtype=_np.complex128)                 #

            for i in range(t_num):
                z[:] = _np.exp(-1j*E*t_all[i])*w
                psi_t[i, :] = phi@z
        elif method == 'Pade':                                          # Pade approximants
            M1 = M.tocsc()

            for i in range(t_num):
                U = _la.expm(-1j*t_all[i]*M1)
                psi_t[i, :] = U@psi0
        elif method == 'emm':                                           # scale and square
            assert _np.std(_np.diff(t_all)) < 1e-6, 'Input t_all is not a linspace-like!'

            psi_t[:, :] = _ssp.linalg.expm_multiply(-1j*M, psi0, start=t_all.min(), stop=t_all.max(), num=t_all.size, endpoint=True)
        else:
            raise ValueError('Unsupported method: ' + method)

#   Finalise
        data = {'t_min': t_min, 't_max': t_max, 't_num': t_num}         #
        data['wf_norm_t'] = _np.array([_la.norm(psi_t[i, :]) for i in range(t_num)])

        self._ue_finalise(psi_t, config, data)                          #

        return data

class LatticeSOModel(LatticeModel):
    """
    Generic tight-binding lattice model with spin-orbit coupling (spin-1/2)
    The spinor wavefunction is flattened in the following way
        Psi[i, mu] = Psi[2*i + mu]
    where
        i   is the site index
        mu  is the spin index: 0 == down, 1 == up
    """
    def _init_local(self):
        """
        Local initialisations
        """
        if 'm' not in dir(self.wf):
            assert len(self.H)%len(self.L) == 0, 'Hamiltonian size is not divisible by lattice size: number of spin components is not defined!'

            m = len(self.H)//len(self.L)                                # number of spin components
            self.wf.m = _np.uint8(m)

    def _sanity_local(self):
        """
        """
        assert 2*len(self.L) == len(self.H), 'Lattice and Hamiltonian sizes mismatch!'

#   ?
    def _ue_finalise(self, psi_t, config, data):
        """
        """
        if 'wf_full_keep' in config:                                    # keep the full unitary evolution
            data['psi_t'] = psi_t
        else:                                                           # only keep the last step of the unitary evolution
            data['psi_max'] = psi_t[-1]

#   ?
        L = self.L                                                      # lattice
        sl = L['sl']                                                    # sublattice structure
        sl_num = L['sl_num']                                            # number of sublattices

        t_num = psi_t.shape[0]                                          # number of timesteps

#   ?
        if 'wf_b' in config:                                            # wavefunction weight on the boundary as a function of time
#            b_ix = _lsm.lattice.generic.boundary_index(L)               #
            b_ix = L.boundary()                                         # indices of the sites on the boundary of the lattice

            b_t = _np.zeros((2, t_num))                                 #
            for i in range(2):
                b_t[i, :] = _np.sum(_np.abs(psi_t[:, 2*b_ix + i])**2, axis=1)
            data['wf_b_t'] = b_t

        if 'wf_index' in config:                                        # sum of probabilities on given subset of the lattice specified by wf_index
            ix = config['wf_index']                                     # indices of the sites in the set

            pi_t = _np.zeros((2, ix.size, t_num))
            for i in range(2):
                pi_t[i, :, :] = _np.transpose(_np.abs(psi_t[:, 2*ix + i])**2)
            data['wf_index_t'] = pi_t

        if 'ipr' in config:                                             # inverse participation ratio (IPR)
            q_all = config['ipr']                                       # the IPR exponents
            q_num = q_all.size                                          # number of the exponents

            data['q_all'] = q_all                                       # compute IPR separately for up/down
            ipr_t = _np.zeros((3, q_num, t_num))                        # spin components: both, up, down
            ipr_t[0, :, :] = _np.array([[_wf.ipr_get(psi_t[i, :], q) for i in range(t_num)] for q in q_all])
            for i in range(2):
                ipr_t[1 + i, :, :] = _np.array([[_wf.ipr_get(psi_t[j, i::2], q) for j in range(t_num)] for q in q_all])
            data['ipr_t'] = ipr_t

#   ?
def psi_lattice_index(ix, m):
    """
    Convert lattice indices into the wavefunction indices in the spinfull
    case. For now only spin-1/2 is supported

    Parameters
    ----------
    ix : array
        the lattice indices
    m : int
        number of spin components
            1       no spin
            2       spin-1/2
            3       spin-1

    Returns
    -------
    ix_psi : array
        the respective wavefunction indices
    """
    if m == 1:
        return ix
    elif m > 1:
        ix_psi = _np.sort(_np.concatenate([ix*m + i for i in range(m)]))
    else:
        raise ValueError('Unsupported number of spin components: ' + str(m))

    return ix_psi

#   CLS search
def cls_from_set(M, E, ix, loop_max=100, eps=_cls_eps, solver='eig'):
    """
    Find compact localised state (CLS) inside a given set ix recasting
    the eigenvalue problem as a fixed point (E = 1) one + no leakage

    Parameters
    ----------
    M : LatticeModel
    E : float
        target flatband eigenenergy
    ix : array
        the set of wavefunction indices
    loop_max : int, optional
        maximal number of iterations
    eps : float, optional
        zero threshold, divergence criterium

    Returns
    -------
    found : bool, optional
        found the CLS
    psi : array
        the final wavefunction
    psi_n : float
        the renormalisation of the norm:
            < 1.0:  no CLS
            1.0 :   CLS?
    cls_ix : array or None
        indices of the cls sites
    """
    H = M['M']                                                          # the Hamiltonian matrix
#    H0 = H.tolil()                                                      #
#    for i in range(H.shape[0]):
#        H0[i, i] += 1.0 - E
#    H0 = H0.tocsr()

    found = False                                                       # success indicator

#   Searching the CLS
    if solver == 'eig':                                                 # use eigendecomposition to find the CLS
        ix_s = _np.sort(ix)                                             #
        ME_ix = _ham.hm_submatrix(H, E, ix, eps=eps, sparse=False)      #

        if _np.all(_np.abs(ME_ix) < eps):                               # zero matrix
            found = False

        (l, phi) = _la.eigh(ME_ix)                                      #
        zs = _np.where(_np.abs(l) < eps)[0]                             # zero modes

        assert zs.size < 2, 'Multiple zero modes discovered on the subset!'

        if sz.size == 0:                                                # no zero modes
            found = False
        else:                                                           # one zero mode
            psi[ix_s] = phi[:, zs[0]]
    elif solver in ('iter', 'iter2', 'ue'):                             # ?
        psi0 = M['psi']                                                 # initial wavefunction
        psi = _np.zeros_like(psi0)                                      # psi
        psi_h = _np.zeros_like(psi0)                                    # H@psi
        psi[ix] = psi0[ix]                                              #
        psi_n = 0.0                                                     # renormalisation coefficient

        if solver == 'iter':                                            # use iterations to find the CLS
            psi_next_func = lambda x: H@x + (1.0 - E)*x                 #
        elif solver == 'iter2':                                         #
            N = H.shape[0]                                              # number of ?

            H1 = H.copy()                                               #
            for i in range(N):                                          # compute H1 = H - E*I
                H1[i, i] -= E

            H1c = H1.copy()
            H1c.data[:] = _np.conjugate(H1c.data)                       # H1c == Hermitian conjugate of H1

            M = H1c.T@H1                                                # M = (H - E*I)^dagger @ (H - E*I)

            (w, v) = ssp.linalg.eigsh(M, k=1, which='LM')               #

            M.data[:] = -M.data[:]/w[0]
            for i in range(N):
                M[i, i] += 1.0

            psi_next_func = lambda x: M@x                               #
        else:                                                           # unitary evolution
            psi_next_func = lambda x: _ssp.linalg.expm_multiply(-1j*H, x, start=0.0, stop=1.0, num=2, endpoint=True)[1]

#   ?
        for loop in range(loop_max):                                    # iterate
            psi_h[:] = psi_next_func(psi)                               #
            psi[ix] = psi_h[ix]                                         #
            psi_n = _la.norm(psi)                                       #
            psi /= psi_n                                                # renormalise

            if _la.norm(H@psi - E*psi) < eps:                           #
                found = True

                break
    else:
        raise ValueError('Unknown solver: ' + solver)

#   Finalise
    if found:                                                           # found a CLS
        cls_ix = _np.where(_np.abs(psi) > eps)[0]                       # indices of the CLS sites
    else:                                                               # not found a CLS
        cls_ix = None

    return (found, psi, psi_n, cls_ix)

def cls_find(M, E, ix, eps=_cls_eps, sh_max=-1, loop_max=100, solver='eig'):
    """
    Find CLS for given flatband

    Parameters
    ----------
    M : LatticeModel
    E : float
        the target flatband energy
    ix : array
        initial guess for the CLS, should be smaller than the actual CLS
    eps : float, optional
    sh_max : int, optional
        maximal shell of ix
    loop_max : int, optional
        maximum number of iterations

    Returns
    -------
    cls : array or None
        the CLS
    cls_ix : array or None
    """
    L = M['L']                                                          # the lattice
    N = len(M)                                                          # number of sites
    m = M['m']                                                          # number of spin components
#    is_spinfull = M.is_spinfull()                                       # SO or no SO?

    sh_ix = _lsm.lattice.generic.shell_index(L, ix, k_max=sh_max)       # (full) shell index

    found = False                                                       # the CLS found flag
    k = 0                                                               # shell counter
    while not found:                                                    # iterate until found a CLS, or no more shells left
        if k > sh_max and sh_max > 0:                                   #
            break

        ix_k = _np.array([i for i in range(N) if sh_ix[i] > -1 and sh_ix[i] <= k])# current CLS set
        ix_cur = M.psi_index(ix_k)                                      # wavefunction set

        r = cls_from_set(M, E, ix_cur, eps=eps, loop_max=loop_max, solver=solver)#

        found = r[0]                                                    #

        k += 1

#   Finalise
    if found:                                                           # found a CLS
        cls = r[1]

        if m == 1:                                                      # no spin
            cls_ix = r[3]
        elif m > 1:                                                     # SO
            cls_ix = _np.array([i for i in ix_k if _np.any([_np.abs(cls[i*m + a]) > eps for a in range(m)])])

        else:
            raise ValueError('Unsupported number of spin components: ' + str(m))
    else:                                                               # not found a CLS
        cls = None
        cls_ix = None

    return (cls, cls_ix)

#   Wavefunction generation
def _wf_circle_make(L, config, eps=1e-4):
    """
    Generate a "circular" wavefunction

    Parameters
    ----------
    L : Lattice
    config : dict
    eps : float, optional
        zero threshold

    Returns
    -------
    data : dict
    """
    r = L['r']                                                          # lattice sites

    r0 = config['point']                                                # the center of the "circle"
    x = config['radius']                                                # the radius of the circle

    R = L.distance_from_point(r0)                                       #

    ix = _np.where(R < x*(1 + eps))[0]                                  # indices of the sites within the circle

    psi = _np.zeros(len(L), dtype=_np.complex128)
    psi[ix] = 1./_mt.sqrt(ix.size)

    data = {'psi': psi, 'index': ix}

    return data

def wf_make(L, wf_type, config):
    """
    Generate a wavefunction

    Parameters
    ----------
    L : Lattice
    wf_type : str
    config : dict

    Returns
    -------
    data : dict
    """
    if wf_type == 'circle':                                             #
        data = _wf_circle_make(L, config)
    elif wf_type == 'circle_graph':
        raise NotImplementedError('Has not been implemented yet!')
    else:
        raise ValueError('Unsupported wavefunction type: ' + wf_type)

    return data
