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

import sys
#@lint-avoid-python-3-compatibility-imports

def get_words(parse):
    return [w.strip(")")
            for w in parse.split()
            if w[-1] == ')']

for l in open(sys.argv[1]):
    splits = l.strip().split("\t")
    if len(splits) != 4:
        continue
    title_parse, article_parse, title, article = splits
    title_words = title.split()
    article_words = article.split()

    # No blanks.
    if any((word == "" for word in title_words)):
        continue

    if any((word == "" for word in article_words)):
        continue

    if not any((word == "." for word in article_words)):
        continue

    # Spurious words to blacklist.
    # First set is words that never appear in input and output
    # Second set is punctuation and non-title words.
    bad_words = ['update#', 'update', 'recasts', 'undated', 'grafs', 'corrects',
                 'retransmitting', 'updates', 'dateline', 'writethru',
                 'recaps', 'inserts', 'incorporates', 'adv##',
                 'ld-writethru', 'djlfx', 'edits', 'byline',
                 'repetition', 'background', 'thruout', 'quotes',
                 'attention', 'ny###', 'overline', 'embargoed', 'ap', 'gmt',
                 'adds', 'embargo',
                 'urgent', '?', ' i ', ' : ', ' - ', ' by ', '-lrb-', '-rrb-']
    if any((bad in title.lower()
            for bad in bad_words)):
        continue

    # Reasonable lengths
    if not (10 < len(article_words) < 100 and
            3 < len(title_words) < 50):
        continue

    # Some word match.
    matches = len(set([w.lower() for w in title_words if len(w) > 3]) &
                  set([w.lower() for w in article_words if len(w) > 3]))
    if matches < 1:
        continue

    # Okay, print.
    print(l.strip())
