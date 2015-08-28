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

-- Ngram neural language model with auxiliary model
require('nn')
require('nngraph')
require('fbnn')
require('cunn')
require('sys')
local utils = require('summary.util')

local nnlm = {}

function nnlm.addOpts()
   cmd:option('-epochs',         5, "Number of epochs to train.")
   cmd:option('-miniBatchSize', 64, "Size of training minibatch.")
   cmd:option('-printEvery', 10000,  "How often to print during training.")
   cmd:option('-modelFilename', '', "File for saving loading/model.")
   cmd:option('-window',         5, "Size of NNLM window.")
   cmd:option('-embeddingDim',  50, "Size of NNLM embeddings.")
   cmd:option('-hiddenSize',   100, "Size of NNLM hiddent layer.")
   cmd:option('-learningRate', 0.1, "SGD learning rate.")
end


function nnlm.create_lm(opt, dict, encoder, encoder_size, encoder_dict)
   local new_mlp = {}
   setmetatable(new_mlp, { __index = nnlm })
   new_mlp.opt = opt
   new_mlp.dict = dict
   new_mlp.encoder_dict = encoder_dict
   new_mlp.encoder_model = encoder
   new_mlp.window = opt.window
   if encoder ~= nil then
      new_mlp:build_mlp(encoder, encoder_size)
   end
   return new_mlp
end


function nnlm:build_mlp(encoder, encoder_size)
   -- Set constants
   local D = self.opt.embeddingDim
   local N = self.opt.window
   local H = self.opt.hiddenSize
   local V = #self.dict.index_to_symbol
   local P = encoder_size
   print(H, P)

   -- Input
   local context_input = nn.Identity()()
   local encoder_input = nn.Identity()()
   local position_input = nn.Identity()()

   local lookup = nn.LookupTable(V, D)(context_input)
   local encoder_node = encoder({encoder_input, position_input, context_input})

   -- tanh W (E y)
   local lm_mlp = nn.Tanh()(nn.Linear(D * N, H)(nn.View(D * N)(lookup)))

   -- Second layer: takes LM and encoder model.
   local mlp = nn.Linear(H + P, V)(nn.View(H + P)(nn.JoinTable(2)(
                                                     {lm_mlp, encoder_node})))
   self.soft_max = nn.LogSoftMax()(mlp)

   -- Input is conditional context and ngram context.
   self.mlp = nn.gModule({encoder_input, position_input, context_input},
      {self.soft_max})

   self.criterion = nn.ClassNLLCriterion()
   self.lookup = lookup.data.module
   self.mlp:cuda()
   self.criterion:cuda()
   collectgarbage()
end


-- Run validation
function nnlm:validation(valid_data)
   print("[Running Validation]")

   local offset = 1000
   local loss = 0
   local total = 0

   valid_data:reset()
   while not valid_data:is_done() do
      local input, target = valid_data:next_batch(offset)
      local out = self.mlp:forward(input)
      local err = self.criterion:forward(out, target) * target:size(1)

      -- Augment counters.
      loss = loss + err
      total = total + target:size(1)
   end
   print(string.format("[perp: %f validation: %f total: %d]",
                       math.exp(loss/total),
                       loss/total, total))
   return loss / total
end


function nnlm:renorm(data, th)
    local size = data:size(1)
    for i = 1, size do
        local norm = data[i]:norm()
        if norm > th then
            data[i]:div(norm/th)
        end
    end
end


function nnlm:renorm_tables()
    -- Renormalize the lookup tables.
    if self.lookup ~= nil then
        print(self.lookup.weight:size())
        print(self.lookup.weight:type())
        self:renorm(self.lookup.weight, 1)
    end
    if self.encoder_model.lookup ~= nil then
        self:renorm(self.encoder_model.lookup.weight, 1)
        if self.encoder_model.title_lookup ~= nil then
            self:renorm(self.encoder_model.title_lookup.weight, 1)
        end
    end
    if self.encoder_model.lookups ~= nil then
        for i = 1, #self.encoder_model.lookups do
            self:renorm(self.encoder_model.lookups[i].weight, 1)
        end
    end
end


function nnlm:run_valid(valid_data)
   -- Run validation.
   if valid_data ~= nil then
      local cur_valid_loss = self:validation(valid_data)
      -- If valid loss does not improve drop learning rate.
      if cur_valid_loss > self.last_valid_loss then
         self.opt.learningRate = self.opt.learningRate / 2
      end
      self.last_valid_loss = cur_valid_loss
   end

   -- Save the model.
   self:save(self.opt.modelFilename)
end


function nnlm:train(data, valid_data)
   -- Best loss seen yet.
   self.last_valid_loss = 1e9
   -- Train
   for epoch = 1, self.opt.epochs do
      data:reset()
      self:renorm_tables()
      self:run_valid(valid_data)

      -- Loss for the epoch.
      local epoch_loss = 0
      local batch = 1
      local last_batch = 1
      local total = 0
      local loss = 0

      sys.tic()
      while not data:is_done() do
         local input, target = data:next_batch(self.opt.miniBatchSize)
         if data:is_done() then break end

         local out = self.mlp:forward(input)
         local err = self.criterion:forward(out, target) * target:size(1)
         local deriv = self.criterion:backward(out, target)

         if not utils.isnan(err) then
            loss = loss + err
            epoch_loss = epoch_loss + err

            self.mlp:zeroGradParameters()
            self.mlp:backward(input, deriv)
            self.mlp:updateParameters(self.opt.learningRate)
         else
            print("NaN")
            print(input)
         end

         -- Logging
         if batch % self.opt.printEvery == 1 then
            print(string.format(
                     "[Loss: %f Epoch: %d Position: %d Rate: %f Time: %f]",
                     loss / ((batch - last_batch) * self.opt.miniBatchSize),
                     epoch,
                     batch * self.opt.miniBatchSize,
                     self.opt.learningRate,
                     sys.toc()
            ))
            sys.tic()
            last_batch = batch
            loss = 0
         end

         batch = batch + 1
         total = total + input[1]:size(1)
      end
      print(string.format("[EPOCH : %d LOSS: %f TOTAL: %d BATCHES: %d]",
                          epoch, epoch_loss / total, total, batch))
   end
end


function nnlm:save(fname)
    print("[saving mlp: " .. fname .. "]")
    torch.save(fname, self)
    return true
end


function nnlm:load(fname)
    local new_self = torch.load(fname)
    for k, v in pairs(new_self) do
       if k ~= 'opt' then
          self[k] = v
       end
    end
    return true
end


return nnlm
