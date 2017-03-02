require 'torch'
require 'nn'

local ra, parent = torch.class('nn.RA', 'nn.Container')
local glimpseUtil = require 'Glimpse'

local function clone_n_times(model, T)
  local t = {}
  for i = 1, T do
    local recurrent = model
    if i > 1 then
      recurrent = model:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    table.insert(t, recurrent)
  end
  return t
end

function ra:__init(rnn, locator, nGlimpses, params)
  parent.__init(self)
  self.T = nGlimpses
  self.H = params[1]
  self.patchSize = params[2]
  self.rnn = rnn
  self.locator = clone_n_times(locator, self.T)
  
  self.modules = {rnn, locator}
  
  self.output = {}
  self.actions = {}
  self.patches = {}
end

function ra:updateOutput(input)
  self.recomputeBackward = true
  self.rnn:forget()
  local N, H, T = input:size(1), self.H, self.T
  self.initalHiddenStates = input.new():resize(N, H):zero()
  local initH = self.initalHiddenStates
  local rnnOutput, l_m = nil, nil
  for t = 1, T do
    if t == 1 then
      l_m = self.locator[t]:forward(initH)
    else
      l_m = self.locator[t]:forward(rnnOutput)
    end
    self.actions[t] = l_m:clone()
    self.patches[t] = glimpseUtil.computePatch(input,l_m, self.patchSize)
    rnnOutput = self.rnn:forward({self.patches[t], l_m})
    self.output[t] = rnnOutput:clone()
  end
  return self.output
end

function ra:backward(input, gradOutput)
  local N, T = input:size(1), self.T
  
  local grad_h, actionInput, gradAction = nil, nil, nil
  self.gradInput:resizeAs(input):zero()
  local dummyGradActionOutput = input.new():resize(N, 2):zero()
  for t = T, 1, -1 do
    if t == T then
      grad_h = gradOutput[t]
    else
      grad_h = gradOutput[t] + gradAction
    end
--    self.gradInput:add(self.rnn:backward({input, self.actions[t]}, grad_h)[1])
    self.rnn:backward({self.patches[t], self.actions[t]}, grad_h)
    if t == 1 then
      actionInput = self.initalHiddenStates
    else
      actionInput = self.output[t-1]
    end
    gradAction = self.locator[t]:backward(actionInput, dummyGradActionOutput)
  end
  return self.gradInput
end

function ra:updateGradInput(input, gradOutput)
  self:backward(input, gradOutput)
  self.recomputeBackward = false
  return self.gradInput
end

function ra:accGradParameters(input, gradOutput)
  if self.recomputeBackward then
    self:backward(input, gradOutput)
  end
end

function ra:reinforce(reward)
  for t = 1, self.T do
    self.locator[t]:reinforce(reward)
  end
end