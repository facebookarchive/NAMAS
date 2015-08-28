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

-- The top-level training script
require('torch')
require('nngraph')

local nnlm = require('summary.nnlm')
local data = require('summary.data')
local encoder  = require('summary.encoder')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a summarization model.')
cmd:text()

data.add_opts(cmd)
encoder.add_opts(cmd)
nnlm.addOpts(cmd)

opt = cmd:parse(arg)

local function main()
   -- Load in the data.
   local tdata = data.load_title(opt.titleDir, true)
   local article_data = data.load_article(opt.articleDir)

   local valid_data = data.load_title(opt.validTitleDir, nil, tdata.dict)
   local valid_article_data =
      data.load_article(opt.validArticleDir, article_data.dict)

   -- Make main LM
   local train_data = data.init(tdata, article_data)
   local valid = data.init(valid_data, valid_article_data)
   local encoder_mlp = encoder.build(opt, train_data)
   local mlp = nnlm.create_lm(opt, tdata.dict, encoder_mlp,
                              opt.bowDim, article_data.dict)

   mlp:train(train_data, valid)
end

main()
