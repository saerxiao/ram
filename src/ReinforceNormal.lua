require 'torch'
require 'nn'

local rn, parent = torch.class('nn.ReNormal', 'nn.Module')

function rn:__init(std)
  parent.__init(self)
  self.std = std
end

function rn:updateOutput(input)
  self.output = input.new():resizeAs(input)
  self.output:normal():mul(self.std):add(input)
  return self.output
end

function rn:updateGradInput(input, gradOutput)
  -- ignore the gradOutput because this is a stochastic step
  -- ref: https://arxiv.org/pdf/1506.05254v3.pdf
  -- the extra nagative sign is because it's reward not loss
  self.gradInput = (self.output - input) / (-self.std * self.std)
  local N = self.reward:size(1)
  if self.reward then
    local reward_expand = self.reward:view(N, 1):expand(N, input:size(2))
    self.gradInput:cmul(reward_expand)
  end
  return self.gradInput
end

function rn:reinforce(reward)
  self.reward = reward
end

