require 'torch'
require 'nn'

local rnn, parent = torch.class('nn.Rnn', 'nn.Module')

local function clone_n_times(container, model, T)
  local t = {}
  for i = 1, T do
    local cloned = model:clone('weight', 'bias', 'gradWeight', 'gradBias')
      table.insert(t, cloned)
      container:add(cloned)
  end
  return t
end

function rnn:__init(hiddenSize, glimpse, recurrent, transfer, nSteps)
  local container = nn.Container()
  self.glimpses = clone_n_times(container, glimpse, nSteps)
  self.rnns = clone_n_times(container, recurrent, nSteps)
  self.T = nSteps
  self.step = 0
end

function rnn:updateOutput(input)
  self.step = self.step + 1
  local step = self.step
  local g = self.glimpses[step]:forward(input)
  local prev_h, prev_c = nil, nil
  if step > 1 then
    prev_h, prev_c = self.rnns[step-1].output, self.rnns[step-1].cell
  end
  local output = self.rnns[step]:forward({g, prev_h, prev_c})
  if step == self.T then
    self.recomputeBackward = true
  end
  
  return output
end

function rnn:backward(input, gradOutput)
  local step = self.step
  local h, c, grad_next_h, grad_next_c = nil, nil, nil,nil
  if step > 1 then
    h, c = self.rnns[step-1].output, self.rnns[step-1].cell
  end
  if step < self.T then 
    grad_next_h= self.rnns[step+1].gradInput[2]
    if #self.rnns[step+1].gradInput == 3 then
      grad_next_c = self.rnns[step+1].gradInput[3]
    end
  end
  local grad_g, grad_h, grad_c = unpack(self.rnns[step]:backward({self.glimpses[step].output, h, c}, {gradOutput, grad_next_h, grad_next_c}))
  local gradInput = self.glimpses[step]:backward(input, grad_g)
  self.step = step - 1
  return gradInput
end

function rnn:updateGradInput(input, gradOutput)
  self.gradInput = self:backward(input, gradOutput)
  if self.step == 0 then
    self.recomputeBackward = false
  end
  return self.gradInput
end

function rnn:accGradParameters(input, gradOutput)
  if self.recomputeBackward then
    self:backward(input, gradOutput)
  end
end

function rnn:forget()
  self.step = 0
end