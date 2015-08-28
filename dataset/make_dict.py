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
from collections import Counter
#@lint-avoid-python-3-compatibility-imports

title_words = Counter()
article_words = Counter()
limit = int(sys.argv[3])

for l in open(sys.argv[1]):
    splits = l.strip().split("\t")
    if len(splits) != 4:
        continue
    title_parse, article_parse, title, article = l.strip().split("\t")
    title_words.update(title.lower().split())
    article_words.update(article.lower().split())

with open(sys.argv[2] + ".article.dict", "w") as f:
    print >>f, "<unk>", 1e5
    print >>f, "<s>", 1e5
    print >>f, "</s>", 1e5
    for word, count in article_words.most_common():
        if count < limit:
            break
        print >>f, word, count

with open(sys.argv[2] + ".title.dict", "w") as f:
    print >>f, "<unk>", 1e5
    print >>f, "<s>", 1e5
    print >>f, "</s>", 1e5
    for word, count in title_words.most_common():
        if count < limit:
            break
        print >>f, word, count
