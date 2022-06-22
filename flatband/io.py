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
    if 'rng_seed' in config:                                            # RNG seed is present
        (rng, rng_seed) = _rio.init(config['rng_seed'])                 # initialise RNG
    else:                                                               # no seed
        rng = None                                                      # for convenience
    config['rng'] = rng                                                 # RNG
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

    return config
