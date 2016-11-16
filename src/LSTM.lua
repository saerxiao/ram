require 'torch'
require 'nn'


local layer, parent = torch.class('nn.LSTM', 'nn.Module')

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

  self.weight = torch.Tensor(D + H, 4 * H)
  self.gradWeight = torch.Tensor(D + H, 4 * H):zero()
  self.bias = torch.Tensor(4 * H)
  self.gradBias = torch.Tensor(4 * H):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (N, H)
  self.gates = torch.Tensor()   -- This will be (N, 4H)
--  self.buffer1 = torch.Tensor() -- This will be (N, H)
--  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (1, 4H)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)

--  self.x = torch.Tensor() -- This will be (N, D)
--
  self.prev_h = torch.Tensor()
  self.prev_c = torch.Tensor()
--  self.remember_states = false

--  self.grad_c_next = torch.Tensor()
--  self.grad_h_next = torch.Tensor()
--  self.grad_x = torch.Tensor()
--  self.gradInput = {self.grad_c_next, self.grad_h_next, self.grad_x}

end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  self.bias:zero()
  self.bias[{{self.hidden_dim + 1, 2 * self.hidden_dim}}]:fill(1)
  self.weight:normal(0, std)
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


function layer:_unpack_input(input)
  local c, h, x = nil, nil, nil
  if torch.type(input) == 'table' and #input == 3 then
    c, h, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    h, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return c, h, x
end


function layer:_get_sizes(input, gradOutput)
  local c0, h0, x = self:_unpack_input(input)
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
  local prev_c, prev_h, x = self:_unpack_input(input)
  local N, D, H = self:_get_sizes(input)

  if not prev_c then
    prev_c = input.new():zeros(N, H)
  end  
  if not prev_h then
    prev_h = input.new():zeros(N,H)
  end
  self.prev_c = prev_c
  self.prev_h = prev_h
  
  local bias_expand = self.bias:view(1, 4 * H):expand(N, 4 * H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]

  local h, c = self.output, self.cell
  h:resize(N, H):zero()
  c:resize(N, H):zero()
  self.gates:resize(N, 4 * H):zero()
  local gates = self.gates
  
  gates:addmm(bias_expand, x, Wx)
  gates:addmm(prev_h, Wh)
  gates[{{}, {1, 3 * H}}]:sigmoid()
  gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
  local i = gates[{{}, {1, H}}]
  local f = gates[{{}, {H + 1, 2 * H}}]
  local o = gates[{{}, {2 * H + 1, 3 * H}}]
  local g = gates[{{}, {3 * H + 1, 4 * H}}]
  h:cmul(i, g)
  c:cmul(f, prev_c):add(h)
  h:tanh(c):cmul(o)
  
  return self.output, self.cell
end


-- gradOutput - N x H
-- grad_x - N x H
function layer:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')
  local prev_c, prev_h, x = self:_unpack_input(input)
  if not prev_c then prev_c = self.prev_c end
  if not prev_h then prev_h = self.prev_h end
  local grad_c_next, grad_h_next, grad_h_this = self:_unpack_input(gradOutput)

--  local grad_prev_c, grad_prev_h, grad_x = self.grad_prev_c, self.grad_prev_h, self.grad_x
  local grad_x = x.new():resizeAs(x):zero()
  local N, D, H = self:_get_sizes(input)

  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

--  grad_prev_h:resizeAs(prev_h):zero()
--  grad_prev_c:resizeAs(prev_c):zero()
--  grad_x:resizeAs(self.x):zero()
  local grad_h = grad_h_this:clone()
  if grad_h_next then
    grad_h:add(grad_h_next)
  end
--  local grad_next_c = self.buffer2:resizeAs(prev_c):zero()
  local grad_c = grad_h_this.new():resizeAs(prev_c):zero()
  if grad_c_next then
    grad_c:add(grad_c_next)
  end
  
  local i = self.gates[{{}, {1, H}}]
  local f = self.gates[{{}, {H + 1, 2 * H}}]
  local o = self.gates[{{}, {2 * H + 1, 3 * H}}]
  local g = self.gates[{{}, {3 * H + 1, 4 * H}}]
  
  local grad_a = self.grad_a_buffer:resize(N, 4 * H):zero()
  local grad_ai = grad_a[{{}, {1, H}}]
  local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
  local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
  local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
  
  local next_h, next_c = self.output, self.cell
  -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
  -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
  -- to compute grad_ao; the other values can be overwritten after we compute
  -- grad_next_c
  local tanh_next_c = grad_ai:tanh(next_c)
  local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
  local my_grad_next_c = grad_ao
  my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_h)
  grad_c:add(my_grad_next_c)
  
  -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
  -- that we can overwrite it.
  grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_h)
  
  -- Use grad_ai as a temporary buffer for computing grad_ag
  local g2 = grad_ai:cmul(g, g)
  grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_c)

  -- We don't need any temporary storage for these so do them last
  grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_c)
  grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_c)
  
  grad_x:mm(grad_a, Wx:t())
  grad_Wx:addmm(scale, x:t(), grad_a)
  grad_Wh:addmm(scale, prev_h:t(), grad_a)
  local grad_a_sum = self.buffer3:resize(1, 4 * H):sum(grad_a, 1)
  grad_b:add(scale, grad_a_sum)

  grad_h:mm(grad_a, Wh:t()) -- grad_h now means (dLdh(t+1))(dh(t+1)dh(t))
  grad_c:cmul(f)
  
--  grad_prev_h:copy(grad_next_h)
--  grad_prev_c:copy(grad_next_c)
  
  self.gradInput = {grad_c, grad_h, grad_x}

  return self.gradInput
end


function layer:clearState()
  self.cell:set()
  self.gates:set()
--  self.buffer1:set()
--  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()

--  self.grad_c0:set()
--  self.grad_h0:set()
--  self.grad_x:set()
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

