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

-- The utility tool box
local util = {}

function util.string_shortfloat(t)
    return string.format('%2.4g', t)
end

function util.shuffleTable(t)
    local rand = math.random
    local iterations = #t
    local j
    for i = iterations, 2, -1 do
       j = rand(i)
       t[i], t[j] = t[j], t[i]
    end
end


function util.string_split(s, c)
   if c==nil then c=' ' end
   local t={}
   while true do
       local f=s:find(c)
       if f==nil then
           if s:len()>0 then
               table.insert(t, s)
           end
           break
       end
       if f > 1 then
          table.insert(t, s:sub(1,f-1))
       end
       s=s:sub(f+1,s:len())
   end
   return t
end


function util.add(tab, key)
   local cur = tab

   for i = 1, #key-1 do
      local new_cur = cur[key[i]]
      if new_cur == nil then
         cur[key[i]] = {}
         new_cur = cur[key[i]]
      end
      cur = new_cur
   end
   cur[key[#key]] = true
end

function util.has(tab, key)
   local cur = tab
   for i = 1, #key do
      cur = cur[key[i]]
      if cur == nil then
         return false
      end
   end
   return true
end

function util.isnan(x)
    return x ~= x
end

return util
