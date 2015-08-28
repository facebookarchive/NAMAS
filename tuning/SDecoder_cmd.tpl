#!/usr/bin/python

import os
import sys

d = {"src" : ,
     "model" : ,
     "title_len" : }

for l in open("SDecoder_cfg.txt"):
    f, val = l.strip().split()
    d[f] = val

cmd = "cd $ABS; th $ABS/summary/run.lua -modelFilename {model} " + \
      "-inputf {src} " + \
      "-length {title_len} -blockRepeatWords -recombine " + \
      "-beamSize 50 " + \
      "-lmWeight {LM}  -unigramBonus {uni} -bigramBonus {bi} " + \
      "-trigramBonus {tri} -lengthBonus {length} -unorderBonus {ooo} " + \
      "-nbest > $ABS/tuning/nbest.out"

os.system(cmd.format(d))
