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

local util     = require('summary.util')

local features = {}

function features.addOpts(cmd)
   cmd:option('-lmWeight',     1.0, "Feature weight for the neural model.")
   cmd:option('-unigramBonus', 0.0, "Feature weight for unigram extraction.")
   cmd:option('-bigramBonus',  0.0, "Feature weight for bigram extraction.")
   cmd:option('-trigramBonus', 0.0, "Feature weight for trigram extraction.")
   cmd:option('-lengthBonus',  0.0, "Feature weight for length.")
   cmd:option('-unorderBonus', 0.0, "Feature weight for out-of-order.")
end

-- Feature positions.
local NNLM = 1
local UNI  = 2
local BI   = 3
local TRI  = 4
local OO   = 5
local LEN  = 6

local kFeat = 6

function features.init(opt, article_to_title)
   local new_features = {}
   setmetatable(new_features, { __index = features })
   new_features.opt = opt
   new_features.num_features = kFeat
   new_features.article_to_title = article_to_title
   return new_features
end

-- Helper: Are words in article.
function features:has_ngram(words)
   return util.has(self.ngrams[#words], words)
end

-- Augment the feature count based on the new word.
function features:compute(f_new, hyp, out_score, y_i1, i)
   local W = self.opt.window

   -- LM Features.
   f_new[NNLM] = f_new[NNLM] + out_score

   if self:has_ngram({y_i1}) then
      f_new[UNI] = f_new[UNI] + 1
   end

   if self:has_ngram({hyp[i+W], y_i1}) then
      f_new[BI] = f_new[BI] + 1
   end

   if self:has_ngram({hyp[i+W-1], hyp[i+W], y_i1}) then
      f_new[TRI] = f_new[TRI] + 1
   end

   if self.ooordered_ngram[hyp[i+W]] ~= nil and
      self.ooordered_ngram[hyp[i+W]][y_i1] ~= nil then
      f_new[OO] = f_new[OO] + 1
   end

   -- Length
   f_new[LEN] = f_new[LEN] + 1
end

-- Augment the score based on the extractive feature values.
function features:add_features(out, beam)
   local W = self.opt.window
   for k = 1, beam:size(1) do

      -- Exact unigram matches.
      for s, _ in pairs(self.ngrams[1]) do
         out[k][s] = out[k][s] + self.opt.unigramBonus
      end

      -- Exact bigram matches.
      if self.ngrams[2][beam[k][W]] ~= nil then
         for s, _ in pairs(self.ngrams[2][beam[k][W]]) do
            out[k][s] = out[k][s] + self.opt.bigramBonus
         end
      end

      -- Exact trigram matches.
      if self.ngrams[3][beam[k][W-1]] ~= nil and
         self.ngrams[3][beam[k][W-1]][beam[k][W]] then
            for s, _ in pairs(self.ngrams[3][beam[k][W-1]][beam[k][W]]) do
               out[k][s] = out[k][s] + self.opt.trigramBonus
            end
      end

      if self.ooordered_ngram[beam[k][W]] ~= nil then
         for s, _ in pairs(self.ooordered_ngram[beam[k][W]]) do
            out[k][s] = out[k][s] + self.opt.unorderBonus
         end
      end
   end
   out:add(self.opt.lengthBonus)
end

-- Precompute extractive table based on the input article.
function features:match_words(START, article)
   self.ooordered_ngram = {}
   local ordered_ngram = {}
   self.ngrams = {{}, {}, {}}
   local hist = {START, START, START, START}

   for j = 1, article:size(1) do
      local tw = self.article_to_title[article[j]]

      -- Does the current word exist in title dict.
      if tw ~= nil then
         for j2 = 1, j do
            local tw2 = self.article_to_title[article[j2]]
            if tw2 ~= nil then
               util.add(ordered_ngram, {tw2, tw})
               if not util.has(ordered_ngram, {tw, tw2}) then
                  util.add(self.ooordered_ngram, {tw, tw2})
               end
            end
         end

         util.add(self.ngrams[1], {tw})
         util.add(self.ngrams[2], {hist[3], tw})
         util.add(self.ngrams[3], {hist[2], hist[3], tw})
      end

      -- Advance window.
      for k = 2, 4 do
         hist[k-1] = hist[k]
      end
      hist[4] = tw
   end
end

return features
