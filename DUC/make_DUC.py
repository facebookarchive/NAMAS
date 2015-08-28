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

"""Construct the DUC test set. """

import sys
import argparse
import glob
import re
import nltk.data
from nltk.tokenize.treebank import TreebankWordTokenizer
#@lint-avoid-python-3-compatibility-imports

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = TreebankWordTokenizer()
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sum_docs', help="Article directory.", type=str)
    parser.add_argument('--year', help="DUC year to process.", type=str)
    parser.add_argument('--result_docs', help="Reference directory.", type=str)
    parser.add_argument('--ref_dir',
                        help="Directory to output the references.", type=str)
    parser.add_argument('--sys_dir',
                        help="Directory to output the references.", type=str)
    parser.add_argument('--article_file',
                        help="File to output the article sentences..", type=str)
    args = parser.parse_args(arguments)

    refs = [open("{0}/task1_ref{1}.txt".format(args.ref_dir, i), "w")
            for i in range(4)]
    article = open(args.article_file, "w")
    prefix = open(args.sys_dir + "/task1_prefix.txt", "w")
    if args.year == "2003":
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    else:
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    files.sort()
    for f in files:
        docset = f.split("/")[-2][:-1].upper()
        name = f.split("/")[-1].upper()

        # Find references.
        if args.year == "2003":
            matches = list(glob.glob("{0}/{1}*.10.*{2}*".format(
                args.result_docs, docset, name)))
        else:
            matches = list(glob.glob("{0}/{1}*{2}*".format(
                args.result_docs, docset, name)))
        matches.sort()
        assert len(matches) == 4, matches
        for i, m in enumerate(matches):
            print >>refs[i], open(m).read().strip()

        # Make input.
        mode = 0
        text = ""
        for l in open(f):
            if l.strip() in ["</P>", "<P>"]:
                continue
            if mode == 1 and l.strip() != "<P>":
                text += l.strip() + " "
            if l.strip() == "<TEXT>":
                mode = 1
        text = " ".join([w for w in text.split() if w[0] != "&"])

        sents = sent_detector.tokenize(text)
        if len(sents) == 0:
            print >>article
            print >>prefix
            continue
        first = sents[0]

        # If the sentence is too short, add the second as well.
        if len(sents[0]) < 130 and len(sents) > 1:
            first = first.strip()[:-1] + " , " + sents[1]

        first = " ".join(tokenizer.tokenize(first.lower()))
        if ")" in first or ("_" in first and args.year == "2003"):
            first = re.split(" ((--)|-|_) ", first, 1)[-1]
        first = first.replace("(", "-lrb-") \
                     .replace(")", "-rrb-").replace("_", ",")
        print >>article, first
        print >>prefix, first[:75]
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
