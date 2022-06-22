#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  io.py
#
#  Copyright 2017 Alexei Andreanov <alexei@pcs.ibs.re.kr>
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


#   I/O

import json as _json

import numpy as _np

import rng_io as _rio

def config_read(fn_c):
    """
    Read configuration file for hopping matrices

    Parameters
    ----------
    fn_c : str
        JSON configuration file

    Returns
    -------
    config : dict
        the configuration
    """
#   Read raw configuration from stream specified by fn_c
    with open(fn_c) as f:
        config = _json.load(f)

#   Finalisation
#   RNG
    if 'rng_seed' in config:                                            # RNG seed is present
        (rng, rng_seed) = _rio.init(config['rng_seed'])                 # initialise RNG
    else:                                                               # no seed
        rng = None                                                      # for convenience
    config['rng'] = rng                                                 # RNG

#   CLS based generators
    if 'fmt' in config:                                                 #
        config['fmt'] = str(config['fmt'])
    if 'H0' in config:                                                  #
        config['H0'] = _np.diag(config['H0'])                           # convert to matrix
    if 'psi1' in config:                                                # right zero mode of H1
        config['psi1'] = _np.array(config['psi1'])                      #
#    if 'real' not in config:                                            #
#        config['real'] = True
    if 'mp' not in config:                                              # multiplicities
        config['mp'] = None

    if 'bipartite' in config and config['fmt'] == 'U2' and 'type' not in config:
        config['type'] = 'full'                                         #

    if 'particular' in config:                                          # generate particular solution to the CLS equations only
        config['particular'] = True
    else:                                                               #
        config['particular'] = False

#   Sequiential LP optimisers
    if 'slp_type' in config:                                            #
        config['slp_type'] = str(config['slp_type'])
    else:                                                               #
        config['slp_type'] = 'det'
    if 'err_max' in config:                                             # maximum convergence error
        config['err_max'] = _np.float64(config['err_max'])
    if 'loop_max' in config:                                            # maximum number of iterations
        config['loop_max'] = _np.uint32(config['loop_max'])
    if 'eig_eps' in config:                                             # eigenvalue threshold
        config['eig_eps'] = _np.float64(config['eig_eps'])
    if 'lp_solver' in config:                                           # LP solver type
        assert config['lp_solver'] in (None, 'glpk'), 'Unsupported LP solver: ' + str(config['lp_solver'])
    if 'E0' in config:                                                  # initial eigenenergy
        config['E0'] = _np.float64(config['E0'])
    if 'E' in config:                                                   # target FB energy
        config['E'] = _np.float64(config['E'])
    if 'dc_bound' in config:                                            # bound on the couplings shifts
        config['dc_bound'] = _np.float64(config['dc_bound'])            #

        assert config['dc_bound'] > 0., 'Negative bound on couplings detected!'
    if 'dE_bound' in config:                                            # bound on the FB energy shift
        if 'E' in config:                                               #
            del config['dE_bound']                                      #
        else:                                                           #
            config['dE_bound'] = _np.float64(config['dE_bound'])        #

            assert config['dE_bound'] > 0., 'Negative bound on FB energy detected!'

    return config

def ue_config_read(fn):
    """
    """
    with open(fn, 'r') as f:
        config = _json.load(f)

    if 'wf_index' in config:
        config['wf_index'] = _np.array(config['wf_index'])
    if 'ipr' in config:
        config['ipr'] = _np.array(config['ipr'])

    return config
