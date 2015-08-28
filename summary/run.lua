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

require('torch')
require('nn')
require('sys')

local nnlm = require('summary.nnlm')
local encoder = require('summary.encoder')
local beam = require('summary.beam_search')
local utils = require('summary.util')

cmd = torch.CmdLine()

beam.addOpts(cmd)

cutorch.setDevice(2)

cmd:option('-modelFilename', '', 'Model to test.')
cmd:option('-inputf',        '', 'Input article files. ')
cmd:option('-nbest',      false, 'Write out the nbest list in ZMert format.')
cmd:option('-length',         15, 'Maximum length of summary.')
opt = cmd:parse(arg)

-- Map the words from one dictionary to another.
local function sync_dicts(dict1, dict2)
   local dict_map = torch.ones(#dict1.index_to_symbol):long()
   for i = 1, #dict1.index_to_symbol do
      local res = dict2.symbol_to_index[dict1.index_to_symbol[i]]
      dict_map[i] = res or 1
   end
   return dict_map
end

-- Apply digit preprocessing.
local function process_word(input_word)
   local word = string.lower(input_word)
   for i = 1, word:len() do
      if word:sub(i, i) >= '0' and word:sub(i, i) <= '9' then
         word = word:sub(1, i-1) .. '#' .. word:sub(i+1)
      end
   end
   return word
end

local function main()
   -- Load in the dictionaries and the input files.
   local mlp = nnlm.create_lm(opt)
   mlp:load(opt.modelFilename)
   local adict = mlp.encoder_dict
   local tdict = mlp.dict

   local dict_map = sync_dicts(adict, tdict)
   local sent_file = assert(io.open(opt.inputf))
   local len = opt.length
   local W = mlp.window
   opt.window = W

   local sent_num = 0
   for line in sent_file:lines() do
      sent_num = sent_num + 1

      -- Add padding.
      local true_line = "<s> <s> <s> " .. line .. " </s> </s> </s>"
      local words = utils.string_split(true_line)

      local article = torch.zeros(#words)
      for j = 1, #words do
         local word = process_word(words[j])
         article[j] = adict.symbol_to_index[word] or
            adict.symbol_to_index["<unk>"]
      end

      -- Run beam search.
      local sbeam = beam.init(opt, mlp.mlp, mlp.encoder_model,
                              dict_map, tdict)
      local results = sbeam:generate(article, len)

      if not opt.nbest then
         if  #results ==  0 then
            io.write("*FAIL*")
         else
            -- Print out in standard format.
            local len, _, output, _ = unpack(results[1])
            local total = 0
            for j = W+2, W+len - 1 do
               local word = tdict.index_to_symbol[output[j]]
               total = total + #word + 1
               io.write(word, " " )
            end
         end
         print("")
      else
         -- Print out an nbest list in Moses/ZMert format.
         for k = 1, #results do
            io.write(sent_num-1, " ||| ")
            local len, score, output, features = unpack(results[k])
            for j = W+2, W+len - 1 do
               io.write(tdict.index_to_symbol[output[j]], " " )
            end
            io.write(" ||| ")
            for f = 1, features:size(1) do
               io.write(features[f], " ")
            end
            io.write(" ||| ", score)
            print("")
         end
      end
   end
end

main()
