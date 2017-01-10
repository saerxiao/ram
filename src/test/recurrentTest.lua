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
  grads:zero()

  local gweight1 = rnn.initialModule.modules[1].weight
  local gweight2 = rnn.sharedClones[1].modules[1].modules[1].weight
  local gweight3 = rnn.sharedClones[2].modules[1].modules[1].weight
  assert(torch.all(torch.eq(gweight1, gweight2)), 'initalModule and the first recurrent module do not share the same inputModule weight')
  assert(torch.all(torch.eq(gweight2, gweight3)), 'the first and second recurrent modules do not share the same inputModule weight')
  
  local ggrad1 = rnn.initialModule.modules[1].gradWeight
  local ggrad2 = rnn.sharedClones[1].modules[1].modules[1].gradWeight
  local ggrad3 = rnn.sharedClones[2].modules[1].modules[1].gradWeight
  assert(torch.all(torch.eq(ggrad1, ggrad2)), 'initalModule and the first recurrent module do not share the same inputModule gradWeight')
  assert(torch.all(torch.eq(ggrad2, ggrad3)), 'the first and second recurrent modules do not share the same inputModule gradWeight')

  ggrad1:add(torch.rand(D, H))  
  assert(torch.all(torch.eq(ggrad1, ggrad2)), 'initalModule and the first recurrent module do not share the same inputModule gradWeight')
  assert(torch.all(torch.eq(ggrad2, ggrad3)), 'the first and second recurrent modules do not share the same inputModule gradWeight')
  
  gweight1:add(ggrad1)
  assert(torch.all(torch.eq(gweight1, gweight2)), 'initalModule and the first recurrent module do not share the same inputModule weight')
  assert(torch.all(torch.eq(gweight2, gweight3)), 'the first and second recurrent modules do not share the same inputModule weight')
  
  local rweight1 = rnn.sharedClones[1].modules[1].modules[2].weight
  local rweight2 = rnn.sharedClones[2].modules[1].modules[2].weight
  assert(torch.all(torch.eq(rweight1, rweight2)), 'the first and second recurrent modules do not share the same feedbackModule weight')
  
  local rgrad1 = rnn.sharedClones[1].modules[1].modules[2].gradWeight
  local rgrad2 = rnn.sharedClones[2].modules[1].modules[2].gradWeight
  assert(torch.all(torch.eq(rgrad1, rgrad2)), 'the first and second recurrent modules do not share the same feedbackModule gradWeight')
  
  rgrad1:add(torch.rand(H, H))
  assert(torch.all(torch.eq(rgrad1, rgrad2)), 'the first and second recurrent modules do not share the same feedbackModule gradWeight')
  rweight1:add(rgrad1)
  assert(torch.all(torch.eq(rweight1, rweight2)), 'the first and second recurrent modules do not share the same feedbackModule weight')
end

recurrentTest.shareWeight()