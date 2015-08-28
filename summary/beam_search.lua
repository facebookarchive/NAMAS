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

-- A beam search decoder
local data     = require('summary.data')
local features = require('summary.features')
local util     = require('summary.util')

local beam = {}
local INF = 1e9

function beam.addOpts(cmd)
   cmd:option('-allowUNK',         false, "Allow generating <unk>.")
   cmd:option('-fixedLength',      true,  "Produce exactly -length words.")
   cmd:option('-blockRepeatWords', false, "Disallow generating a word twice.")
   cmd:option('-lmWeight',           1.0, "Weight for main model.")
   cmd:option('-beamSize',           100, "Size of the beam.")
   cmd:option('-extractive',       false, "Force fully extractive summary.")
   cmd:option('-abstractive',      false, "Force fully abstractive summary.")
   cmd:option('-recombine',        false, "Used hypothesis recombination.")
   features.addOpts(cmd)
end

function beam.init(opt, mlp, aux_model, article_to_title, dict)
   local new_beam = {}
   setmetatable(new_beam, { __index = beam })
   new_beam.opt = opt
   new_beam.K   = opt.beamSize
   new_beam.mlp = mlp
   new_beam.aux_model = aux_model
   new_beam.article_to_title = article_to_title
   new_beam.dict = dict

   -- Special Symbols.
   new_beam.UNK   = dict.symbol_to_index["<unk>"]
   new_beam.START = dict.symbol_to_index["<s>"]
   new_beam.END   = dict.symbol_to_index["</s>"]

   return new_beam
end

-- Helper: convert flat index to matrix.
local function flat_to_rc(v, indices, flat_index)
   local row = math.floor((flat_index - 1) / v:size(2)) + 1
   return row, indices[row][(flat_index - 1) % v:size(2) + 1]
end

-- Helper: find kmax of vector.
local function find_k_max(pool, mat)
   local v = pool:forward(mat:t()):t()
   local orig_indices = pool.indices:t():add(1)
   return v:contiguous(), orig_indices
end

-- Use beam search to generate a summary of
-- the article of length <= len.
function beam:generate(article, len)
   local n = len
   local K = self.K
   local W = self.opt.window

   -- Initialize the extractive features.
   local feat_gen = features.init(self.opt, self.article_to_title)
   feat_gen:match_words(self.START, article)
   local F = feat_gen.num_features
   local FINAL_VAL = 1000

   -- Initilize the charts.
   -- scores[i][k] is the log prob of the k'th hyp of i words.
   -- hyps[i][k] contains the words in k'th hyp at
   --          i word (left padded with W <s>) tokens.
   -- feats[i][k][f] contains the feature count of
   --               the f features for the k'th hyp at word i.
   local result = {}
   local scores = torch.zeros(n+1, K):float()
   local hyps = torch.zeros(n+1, K, W+n+1):long()
   local feats = torch.zeros(n+1, K, F):float()
   hyps:fill(self.START)

   -- Initilialize used word set.
   -- words_used[i][k] is a set of the words used in the i,k hyp.
   local words_used = {}
   if self.opt.blockRepeatWords then
      for i = 1, n + 1 do
         words_used[i] = {}
         for k = 1, K do
            words_used[i][k] = {}
         end
      end
   end

   -- Find k-max columns of a matrix.
   -- Use 2*k in case some are invalid.
   local pool = nn.TemporalKMaxPooling(2*K)

   -- Main loop of beam search.
   for i = 1, n do
      local cur_beam = hyps[i]:narrow(2, i+1, W)
      local cur_K = K

      -- (1) Score all next words for each context in the beam.
      --    log p(y_{i+1} | y_c, x) for all y_c
      local input = data.make_input(article, cur_beam, cur_K)
      local model_scores = self.mlp:forward(input)

      local out = model_scores:clone():double()
      out:mul(self.opt.lmWeight)

      -- If length limit is reached, next word must be end.
      local finalized = (i == n) and self.opt.fixedLength
      if finalized then
         out[{{}, self.END}]:add(FINAL_VAL)
      else
         -- Apply hard constraints.
         out[{{}, self.START}] = -INF
         if not self.opt.allowUNK then
            out[{{}, self.UNK}] = -INF
         end
         if self.opt.fixedLength then
            out[{{}, self.END}] = -INF
         end

         -- Add additional extractive features.
         feat_gen:add_features(out, cur_beam)
      end

      -- Only take first row when starting out.
      if i == 1 then
         cur_K = 1
         out = out:narrow(1, 1, 1)
         model_scores = model_scores:narrow(1, 1, 1)
      end

      -- Prob of summary is log p + log p(y_{i+1} | y_c, x)
      for k = 1, cur_K do
         out[k]:add(scores[i][k])
      end

      -- (2) Retain the K-best words for each hypothesis using GPU.
      -- This leaves a KxK matrix which we flatten to a K^2 vector.
      local max_scores, mat_indices = find_k_max(pool, out:cuda())
      local flat = max_scores:view(max_scores:size(1)
                                      * max_scores:size(2)):float()

      -- 3) Construct the next hypotheses by taking the next k-best.
      local seen_ngram = {}
      for k = 1, K do
         for _ = 1, 100 do

            -- (3a) Pull the score, index, rank, and word of the
            -- current best in the table, and then zero it out.
            local score, index = flat:max(1)
            if finalized then
               score[1] = score[1] - FINAL_VAL
            end
            scores[i+1][k] = score[1]
            local prev_k, y_i1 = flat_to_rc(max_scores, mat_indices, index[1])
            flat[index[1]] = -INF

            -- (3b) Is this a valid next word?
            local blocked = (self.opt.blockRepeatWords and
                                words_used[i][prev_k][y_i1])

            blocked = blocked or
               (self.opt.extractive and not feat_gen:has_ngram({y_i1}))
            blocked = blocked or
               (self.opt.abstractive and feat_gen:has_ngram({y_i1}))

            -- Hypothesis recombination.
            local new_context = {}
            if self.opt.recombine then
               for j = i+2, i+W do
                  table.insert(new_context, hyps[i][prev_k][j])
               end
               table.insert(new_context, y_i1)
               blocked = blocked or util.has(seen_ngram, new_context)
            end

            -- (3c) Add the word, its score, and its features to the
            -- beam.
            if not blocked then
               -- Update tables with new hypothesis.
               for j = 1, i+W do
                  local pword = hyps[i][prev_k][j]
                  hyps[i+1][k][j] = pword
                  words_used[i+1][k][pword] = true
               end
               hyps[i+1][k][i+W+1] = y_i1
               words_used[i+1][k][y_i1] = true

               -- Keep track of hypotheses seen.
               if self.opt.recombine then
                  util.add(seen_ngram, new_context)
               end

               -- Keep track of features used (For MERT)
               feats[i+1][k]:copy(feats[i][prev_k])
               feat_gen:compute(feats[i+1][k], hyps[i+1][k],
                                model_scores[prev_k][y_i1], y_i1, i)

               -- If we have produced an END symbol, push to stack.
               if y_i1 == self.END then
                  table.insert(result, {i+1, scores[i+1][k],
                                        hyps[i+1][k]:clone(),
                                        feats[i+1][k]:clone()})
                  scores[i+1][k] = -INF
               end
               break
            end
         end
      end
   end

   -- Sort by score.
   table.sort(result, function (a, b) return a[2] > b[2] end)

   -- Return the scores and hypotheses at the final stage.
   return result
end


return beam
