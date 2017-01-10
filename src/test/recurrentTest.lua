package.path = package.path .. ";/home/saxiao/ram/src/?.lua"

require 'torch'
require 'nn'
require 'dp'
require 'rnn'
require 'Recurrent'

local recurrentTest = {}

function recurrentTest.shareWeight()
  local D, H, T = 4, 5, 3
  local glimpse = nn.Linear(D, H)
  local recurrent = nn.Linear(H, H)
  local rnn = nn.Recurrent1(H, glimpse, recurrent, nn.ReLU(), T)
  
  local params, grads = rnn:getParameters()
  params:uniform(-0.1, 0.1)

  local weight1 = rnn.initialModule.modules[1].weight
  local weight2 = rnn.sharedClones[1].modules[1].modules[1].weight
  local weight3 = rnn.sharedClones[2].modules[1].modules[1].weight
  print(weight1, weight2, weight3)
  assert(torch.all(torch.eq(weight1, weight2)), 'initalModule and the first recurrent module do not share the same inputModule')
  assert(torch.all(torch.eq(weight2, weight3)), 'the first and second recurrent modules do not share the same inputModule')
  local x = torch.rand(D, H)
  weight1:add(x)
  assert(torch.all(torch.eq(weight1, weight2)), 'initalModule and the first recurrent module do not share the same inputModule')
  assert(torch.all(torch.eq(weight2, weight2)), 'the first and second recurrent modules do not share the same inputModule')
  print(weight1, weight2, weight3)
--  
--  assert(torch.pointer(rnn.initialModule.modules[1].weight) == torch.pointer(rnn.sharedClones[1].modules[1].modules[1].weight), 
--  'initalModule and the first recurrent module do not share the same inputModule')
--  assert(torch.pointer(rnn.sharedClones[1].modules[1].modules[1].weight) == torch.pointer(rnn.sharedClones[2].modules[1].modules[1].weight), 
--  'the first and second recurrent modules do not share the same inputModule')
end

recurrentTest.shareWeight()