require 'torch'
require 'nn'


local layer, parent = torch.class('nn.VanillaRnn', 'nn.Module')

--[[
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceLSTM stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]

function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H

  self.weight = torch.Tensor(D + H, H)
  self.gradWeight = torch.Tensor(D + H, H):zero()
  self.bias = torch.Tensor(H)
  self.gradBias = torch.Tensor(H):zero()
  self:reset()

  self.prev_h = torch.Tensor()
end


function layer:reset(std)
--  if not std then
--    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
--  end
--  self.bias:zero()
--  self.weight:normal(0, std)
  self.weight:uniform(-0.1, 0.1)
  self.bias:uniform(-0.1, 0.1)
  return self
end


--function layer:resetStates()
--  self.h0 = self.h0.new()
--  self.c0 = self.c0.new()
--end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d, x:size(i) .. ' and ' .. d .. ' should be equal')
  end
end

function layer:_get_sizes(input, gradOutput)
  local x, h0, c0 = unpack(input)
  local N = x:size(1)
  local H, D = self.hidden_dim, self.input_dim
  check_dims(x, {N, D})
  if h0 then
    check_dims(h0, {N, H})
  end
  if c0 then
    check_dims(c0, {N, H})
  end
  if gradOutput then
    check_dims(gradOutput, {N, H})
  end
  return N, D, H
end


--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input, (N, D)

Output:
- h: hidden states, (N, H)
--]]
-- input - N x D
function layer:updateOutput(input)
  self.recompute_backward = true
  local x, prev_h = unpack(input)
  local N, D, H = self:_get_sizes(input)
 
  if not prev_h then
    prev_h = x.new():zeros(N,H)
  end
  self.prev_h = prev_h
  
  local bias_expand = self.bias:view(1, H):expand(N, H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]

  local h = self.output
  h:resize(N, H):zero()
  
  local gate = x.new():resize(N, H):zero()
  gate:addmm(bias_expand, x, Wx)
  gate:addmm(prev_h, Wh)
  h:tanh(gate)
  
  return self.output
end


-- gradOutput - N x H
-- grad_x - N x H
function layer:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')
  local x, prev_h = unpack(input)
  if not prev_h then prev_h = self.prev_h end
  local grad_h_this, grad_h_next = unpack(gradOutput)

  local grad_x = x.new():resizeAs(x):zero()
  local N, D, H = self:_get_sizes(input)

  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  local grad_h = grad_h_this:clone()
  if grad_h_next then
    grad_h:add(grad_h_next)
  end

  local next_h = self.output
  local h2 = torch.cmul(next_h, next_h)
  local grad_u = x.new():resizeAs(next_h):zero()
  grad_u:fill(1):add(-1, h2):cmul(grad_h)
  
  grad_x:mm(grad_u, Wx:t())
  grad_Wx:addmm(scale, x:t(), grad_u)
  grad_Wh:addmm(scale, prev_h:t(), grad_u)
  local grad_u_sum = x.new():resize(1, H):sum(grad_u, 1)
  grad_b:add(scale, grad_u_sum)

  grad_h:mm(grad_u, Wh:t()) -- grad_h now means (dLdh(t+1))(dh(t+1)dh(t))
  
  self.gradInput = {grad_h, grad_x}

  return self.gradInput
end


function layer:clearState()
  self.cell:set()
  self.gates:set()
  self.buffer3:set()
  self.grad_a_buffer:set()

  self.output:set()
end


function layer:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end


function layer:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end


function layer:__tostring__()
  local name = torch.type(self)
  local din, dout = self.input_dim, self.hidden_dim
  return string.format('%s(%d -> %d)', name, din, dout)
end

