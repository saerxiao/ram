require 'torch'
require 'nn'

local ra, parent = torch.class('nn.RA', 'nn.Module')

function ra:__init(rnn, locator, nGlimpses, params)
  self.rnn = rnn
  self.locator = locator
  self.T = nGlimpses
  self.H = params[1]
  
  self.output = {}
  self.actions = {}
end

function ra:updateOutput(input)
  self.rnn:forget()
  self.locator:forget()
  local N, H, T = input:size(1), self.H, self.T
  self.initalHiddenStates = input.new():resize(N, H):zero()
  local initH = self.initalHiddenStates
  local rnnOutput, l_m = nil, nil
  for t = 1, T do
    if t == 1 then
      l_m = self.locator:forward(initH)
    else
      l_m = self.locator:forward(rnnOutput)
    end
    table.insert(self.actions, l_m:clone())
    rnnOutput = self.rnn:forward({input, l_m})
    table.insert(self.output, rnnOutput:clone())
  end
  return self.output
end

function ra:backward(input, gradOutput)
  local N, T = input:size(1), self.T
  
  local grad_h, actionInput, gradInput = nil, nil, nil
  local dummyGradActionOutput = input.new():resize(N, 2):zero()
  for t = T, 1, -1 do
    if t == T then
      grad_h = gradOutput[t]
    else
      if t == 1 then
        actionInput = self.initalHiddenStates
      else
        actionInput = self.output[t-1]
      end
      grad_h = gradOutput[t] + self.locator:backward(actionInput, dummyGradActionOutput)
    end
    gradInput = self.rnn:backward({input, self.actions[t]}, grad_h)
  end
  self.gradInput = gradInput
  return gradInput
end

function ra:reinforce(reward)
  self.locator:reinforce(reward)
end