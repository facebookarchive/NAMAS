#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>

import os
import sys
#@lint-avoid-python-3-compatibility-imports

d = {"src": sys.argv[1],
     "model": sys.argv[2],
     "title_len": 14}

for l in open("tuning/blank.params"):
    f, val = l.strip().split()
    d[f] = val

cmd = "cd $ABS; $CUTH $ABS/summary/run.lua -modelFilename {model} " + \
      "-inputf {src} -recombine " + \
      "-length {title_len} -blockRepeatWords " + \
      "-lmWeight {LM} -unigramBonus {uni} -bigramBonus {bi} " + \
      "-trigramBonus {tri} -lengthBonus {length} -unorderBonus {ooo} "

os.system(cmd.format(**d))
