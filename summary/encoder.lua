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

local encoder = {}

function encoder.add_opts(cmd)
   cmd:option('-encoderModel', 'bow', "The encoder model to use.")
   cmd:option('-bowDim',      50, "Article embedding size.")
   cmd:option('-attenPool',    5, "Attention model pooling size.")
   cmd:option('-hiddenUnits', 1000, "Conv net encoder hidden units.")
   cmd:option('-kernelWidth', 5,    "Conv net encoder kernel width.")
end


function encoder.build(opt, data)
   torch.setdefaulttensortype("torch.CudaTensor")
   local model = nil
   if opt.encoderModel == "none" then
      model = encoder.build_blank_model(opt, data)
   elseif opt.encoderModel == "bow" then
      model =  encoder.build_bow_model(opt, data)
   elseif opt.encoderModel == "attenbow" then
      model = encoder.build_attnbow_model(opt, data)
   elseif opt.encoderModel == "conv" then
      model = encoder.build_conv_model(opt, data)
   end
   torch.setdefaulttensortype("torch.DoubleTensor")
   return model
end


function encoder.build_blank_model(opt, data)
   -- Ignores the article layer entirely (acts like LM).
   local lookup = nn.Identity()()
   local ignore1 = nn.Identity()()
   local ignore2 = nn.Identity()()
   local start = nn.SelectTable(3)({lookup, ignore1, ignore2})

   local mout = nn.MulConstant(0)(start)
   local encoder_mlp = nn.gModule({lookup, ignore1, ignore2}, {mout})
   encoder_mlp:cuda()
   return encoder_mlp
end


function encoder.build_bow_model(opt, data)
   print("Encoder model: Bag-of-Words")

   -- BOW with mean on article.
   local lookup = nn.LookupTable(
      #data.article_data.dict.index_to_symbol,
      opt.bowDim)()

   -- Ignore the context.
   local ignore1 = nn.Identity()()
   local ignore2 = nn.Identity()()

   -- Ignores the context and position input.
   local start = nn.SelectTable(1)({lookup, ignore1, ignore2})
   local mout = nn.Linear(opt.bowDim, opt.bowDim)(
      nn.Mean(3)(nn.Transpose({2, 3})(start)))

   local encoder_mlp = nn.gModule({lookup, ignore1, ignore2}, {mout})
   encoder_mlp:cuda()

   return encoder_mlp
end


function encoder.build_conv_model(opt, data)
   -- Three layer thin convolutional architecture.
   print("Encoder model: Conv")
   local V2 = #data.article_data.dict.index_to_symbol
   local nhid = opt.hiddenUnits

   -- Article embedding.
   local article_lookup = nn.LookupTable(V2, nhid)()

   -- Ignore the context.
   local ignore1 = nn.Identity()()
   local ignore2 = nn.Identity()()
   local start = nn.SelectTable(1)({article_lookup, ignore1, ignore2})
   local kwidth = opt.kernelWidth
   local model = nn.Sequential()
   model:add(nn.View(1, -1, nhid):setNumInputDims(2))
   model:add(cudnn.SpatialConvolution(1, nhid, nhid, kwidth, 1, 1, 0))
   model:add(cudnn.SpatialMaxPooling(1, 2, 1, 2))
   model:add(nn.Threshold())
   model:add(nn.Transpose({2,4}))

   -- layer 2
   model:add(cudnn.SpatialConvolution(1, nhid, nhid, kwidth, 1, 1, 0))
   model:add(nn.Threshold())
   model:add(nn.Transpose({2,4}))

   -- layer 3
   model:add(cudnn.SpatialConvolution(1, nhid, nhid, kwidth, 1, 1, 0))
   model:add(nn.View(nhid, -1):setNumInputDims(3))
   model:add(nn.Max(3))
   local done = nn.View(opt.hiddenUnits)(model(start))

   local mout = nn.Linear(opt.hiddenUnits, opt.embeddingDim)(done)

   local encoder_mlp = nn.gModule({article_lookup, ignore1, ignore2}, {mout})
   encoder_mlp.lookup = article_lookup.data.module
   encoder_mlp:cuda()
   return encoder_mlp
end


function encoder.build_attnbow_model(opt, data)
   print("Encoder model: BoW + Attention")

   local D2 = opt.bowDim
   local N = opt.window
   local V = #data.title_data.dict.index_to_symbol
   local V2 = #data.article_data.dict.index_to_symbol

   -- Article Embedding.
   local article_lookup = nn.LookupTable(V2, D2)()

   -- Title Embedding.
   local title_lookup = nn.LookupTable(V, D2)()

   -- Size Lookup
   local size_lookup = nn.Identity()()

   -- Ignore size lookup to make NNGraph happy.
   local article_context = nn.SelectTable(1)({article_lookup, size_lookup})

   -- Pool article
   local pad = (opt.attenPool - 1) / 2
   local article_match = article_context

   -- Title context embedding.
   local title_context = nn.View(D2, 1)(
      nn.Linear(N * D2, D2)(nn.View(N * D2)(title_lookup)))

   -- Attention layer. Distribution over article.
   local dot_article_context = nn.MM()({article_match,
                                        title_context})

   -- Compute the attention distribution.
   local non_linearity = nn.SoftMax()
   local attention = non_linearity(nn.Sum(3)(dot_article_context))

   local process_article =
      nn.Sum(2)(nn.SpatialSubSampling(1, 1, opt.attenPool)(
                   nn.SpatialZeroPadding(0, 0, pad, pad)(
                      nn.View(1, -1, D2):setNumInputDims(2)(article_context))))

   -- Apply attention to the subsampled article.
   local mout = nn.Linear(D2, D2)(
      nn.Sum(3)(nn.MM(true, false)(
                   {process_article,
                    nn.View(-1, 1):setNumInputDims(1)(attention)})))

   -- Apply attention
   local encoder_mlp = nn.gModule({article_lookup, size_lookup, title_lookup},
      {mout})

   encoder_mlp:cuda()
   encoder_mlp.lookup = article_lookup.data.module
   encoder_mlp.title_lookup = title_lookup.data.module
   return encoder_mlp
end

return encoder
