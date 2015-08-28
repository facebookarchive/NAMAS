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

"""Prep ROUGE eval. """

import sys
import glob
import os
import argparse
import itertools
#@lint-avoid-python-3-compatibility-imports

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=
                                 argparse.RawDescriptionHelpFormatter)
parser.add_argument('--base', help="Base directory.", type=str)
parser.add_argument('--gold', help="Base directory.", type=str)
parser.add_argument('--system', help="Base directory.", type=str)
parser.add_argument('--input', help="Input text.", type=str)

args = parser.parse_args(sys.argv[1:])

for f in glob.glob("{0}/references/*".format(args.base)):
    task, ref = f.split("/")[-1].split("_")
    ref = int(ref.split(".")[0][-1])

    for i, l in enumerate(open(f)):
        os.system("mkdir -p %s/%s%04d"%(args.gold, task, i))
        with open("%s/%s%04d/%s%04d.%04d.gold" % (args.gold, task, i, task, i, ref), "w") as out:
            print >>out, l.strip()


for f in glob.glob("{0}/system/*".format(args.base)):
    task, ref = f.split("/")[-1].split("_", 1)
    #if ref.startswith("ducsystem"): continue
    system = ref.split(".")[0]
    os.system("mkdir -p %s/%s"%(args.system, system))
    for i, (l, input_line) in enumerate(itertools.izip(open(f), open(args.input))):
        words = []
        numbers = dict([(len(w), w) for w in input_line.strip().split() if w[0].isdigit()])
        for w in l.strip().split():
            # Replace # with numbers from the input.
            if w[0] == "#" and len(w) in numbers:
                words.append(numbers[len(w)])
            elif w == "<s>":
                continue
            else:
                words.append(w)

        with open("%s/%s/%s%04d.%s.system" % (args.system, system, task, i, system),"w") as out:
            if words:
                print >>out, " ".join(words)
            else:
                print >>out, "fail"
