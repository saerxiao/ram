require 'torch'
require 'nn'

local rescale, parent = torch.class('nn.Rescale', 'nn.Module')

function rescale:__init(outputSize, min, max)
  parent.__init(self)
  self.min = min and min or 0
  self.max = max and max or 1
  self.dmin = torch.Tensor(outputSize)
  self.dmax = torch.Tensor(outputSize)
end

function rescale:updateOutput(input)
  local N, outputSize = input:size(1), input:size(2)
  assert(outputSize == self.dmin:size(), "input size not matched")
  local min, max, dmin, dmax = self.min, self.max, self.dmin, self.dmax
  local range = max - min
  dmin = input:min(1)
  dmax = input:max(1)
  local drange = dmax - dmin

  local output = input.new():resizeAs(input):copy(input)
  output:add(-dmin:expand(N,outputSize))
  output:mul(range)
  output:mul(1/drange)
  output:add(min)
  self.output = output
  return self.output
end

function rescale:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):copy(gradOutput)
  self.gradInput:mul()
end