--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Alexander M Rush <srush@seas.harvard.edu>
--          Sumit Chopra <spchopra@fb.com>
--          Jason Weston <jase@fb.com>

-- Script to build the dictionary
local utils = require('summary/util')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Build torch serialized version of a dictionary file.')
cmd:text()
cmd:text('Options')
cmd:option('-inf', '', 'The input dictionary.')
cmd:option('-outf', '', 'The output directory.')
cmd:text()

opt = cmd:parse(arg)

local f = io.open(opt.inf, 'r')
local word_id = 0
local dict = {symbol_to_index = {},
              index_to_symbol = {}}
for l in f:lines() do
   word_id = word_id + 1
   local word = utils.string_split(l)[1]
   dict.symbol_to_index[word] = word_id
   dict.index_to_symbol[word_id] = word
end
torch.save(opt.outf, dict)
