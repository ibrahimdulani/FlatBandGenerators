#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  setup.py
#
#  Copyright 2016 alexei andreanov <alexei.andreanov@gmail.com>
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


#

import datetime
from setuptools import setup, find_packages

now = datetime.datetime.now()                                           # current date
vn = '.'.join(map(str, [now.year, now.month, now.day]))                 # version = compilation date

setup(
    name='flatband',                                                    # name of the package
    version=vn,                                                         # version of the package
    description='',
    long_description=open('README.txt').read(),
#    url='http://pypi.python.org/pypi/TowelStuff/',
    author='Alexei Andreanov',                                          # author of the package
    author_email='alexei@pcs.ibs.re.kr',                                # email of the author
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Physicists',
        'Topic :: Flatbands',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='Flatbands',
    package_dir = {'flatband': 'src', 'flatband/model': 'src/model'},
    packages=('flatband','flatband/model'),
    install_requires=[
        'numpy >= 1.12',
        'scipy >= 1.0.0',
        'fake',
        'lsm'
    ],
    test_suite='tests/qn.py'
)
