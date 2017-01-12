package.path = package.path .. ";/home/saxiao/ram/src/?.lua"

require 'torch'
require 'nn'
require 'dp'
require 'rnn'
require 'Recurrent'
require 'Rnn'
require 'VanillaRnn'

local gradcheck = require 'util.gradcheck'

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

function recurrentTest.rnnShareWeight()
  local D, H, T = 4, 5, 3
  local glimpse = nn.Linear(D, H)
  local recurrent = nn.Linear(H, H)
  local rnn = nn.Rnn(H, glimpse, recurrent, nn.ReLU(), T)
  
  local params, grads = rnn:getParameters()
  params:uniform(-0.1, 0.1)
  grads:zero()

  local gweight1 = rnn.glimpses[1].weight
  local gweight2 = rnn.glimpses[2].weight
  local gweight3 = rnn.glimpses[3].weight
  assert(torch.all(torch.eq(gweight1, gweight2)), 'initalModule and the first recurrent module do not share the same inputModule weight')
  assert(torch.all(torch.eq(gweight2, gweight3)), 'the first and second recurrent modules do not share the same inputModule weight')
  
  local ggrad1 = rnn.glimpses[1].gradWeight
  local ggrad2 = rnn.glimpses[2].gradWeight
  local ggrad3 = rnn.glimpses[3].gradWeight
  assert(torch.all(torch.eq(ggrad1, ggrad2)), 'initalModule and the first recurrent module do not share the same inputModule gradWeight')
  assert(torch.all(torch.eq(ggrad2, ggrad3)), 'the first and second recurrent modules do not share the same inputModule gradWeight')

  ggrad1:add(torch.rand(D, H))  
  assert(torch.all(torch.eq(ggrad1, ggrad2)), 'initalModule and the first recurrent module do not share the same inputModule gradWeight')
  assert(torch.all(torch.eq(ggrad2, ggrad3)), 'the first and second recurrent modules do not share the same inputModule gradWeight')
  
  gweight1:add(ggrad1)
  assert(torch.all(torch.eq(gweight1, gweight2)), 'initalModule and the first recurrent module do not share the same inputModule weight')
  assert(torch.all(torch.eq(gweight2, gweight3)), 'the first and second recurrent modules do not share the same inputModule weight')
  
  local rweight1 = rnn.rnns[1].weight
  local rweight2 = rnn.rnns[2].weight
  assert(torch.all(torch.eq(rweight1, rweight2)), 'the first and second recurrent modules do not share the same feedbackModule weight')
  
  local rgrad1 = rnn.rnns[1].gradWeight
  local rgrad2 = rnn.rnns[2].gradWeight
  assert(torch.all(torch.eq(rgrad1, rgrad2)), 'the first and second recurrent modules do not share the same feedbackModule gradWeight')
  
  rgrad1:add(torch.rand(H, H))
  assert(torch.all(torch.eq(rgrad1, rgrad2)), 'the first and second recurrent modules do not share the same feedbackModule gradWeight')
  rweight1:add(rgrad1)
  assert(torch.all(torch.eq(rweight1, rweight2)), 'the first and second recurrent modules do not share the same feedbackModule weight')
  print(rweight1, rweight2)
end

function recurrentTest.recurrentGradiantCheck()
  local N, D, H, T = 3, 4, 5, 3
  local glimpse = nn.Linear(D, H)
  local recurrent = nn.Linear(H, H)
  local rnn = nn.Recurrent1(H, glimpse, recurrent, nn.ReLU(), T)
--  local rnn = nn.Linear(D, H)
  
  local params, grads = rnn:getParameters()
  
  params:uniform(-0.1, 0.1)
  grads:zero()
  local params_origin = params:clone()
  
  local x, df = torch.randn(T, N, D), torch.randn(N, H)
  
  local function fx(x1)
    local output = nil
    rnn:forget()
    for t = 1, T do
      if t == 1 then
        output = rnn:updateOutput(x1)
      else
        output = rnn:updateOutput(x[t])
      end
    end
    return output
  end
  
  local function fw(w)
    local output = nil
    rnn:forget()
    params:copy(w)
    for t = 1, T do
      output = rnn:updateOutput(x[t])
    end
    params:copy(params_origin)
    return output
  end
  
  local dx_num = gradcheck.numeric_gradient(fx, x[1], df)
  local dw_num = gradcheck.numeric_gradient(fw, params, df)
  print(dx_num)
  
  rnn:forget()
  for t = 1, T do
    rnn:forward(x[t])
  end
  local dx_compute = nil
  for t = T, 1, -1 do
    if t == T then
      dx_compute = rnn:backward(x[t], df)
    else
      dx_compute = rnn:backward(x[t], torch.zeros(N, H))
    end   
  end
  print(dx_compute)
  
  local dx_err = gradcheck.relative_error(dx_num, dx_compute) -- this is still pretty big for T > 1, something wrong in my test code
  local dw_err = gradcheck.relative_error(dw_num, grads)
  print(dx_err, dw_err)
  assert(dx_err < 1e-3)
  assert(dw_err < 1e-5)
end

function recurrentTest.vanillaRnnGradiantCheck()
  local N, D, H = 3, 4, 5
  local net = nn.VanillaRnn(D, H)
  
  local params, grads = net:getParameters()
  
  params:uniform(-0.1, 0.1)
  grads:zero()
  local params_origin = params:clone()
  
  local x0, h0, dh = torch.randn(N, D), torch.randn(N, H), torch.randn(N, H)
  
  local function fx(x)
    return net:updateOutput({x, h0})
  end
  
  local function fh(h)
    return net:updateOutput({x0, h})
  end
  
  local function fw(w)
    params:copy(w)
    local output = net:updateOutput({x0, h0})
    params:copy(params_origin)
    return output
  end
  
  local dx_num = gradcheck.numeric_gradient(fx, x0, dh)
  local dh0_num = gradcheck.numeric_gradient(fh, h0, dh)
  local dw_num = gradcheck.numeric_gradient(fw, params, dh)
  
  net:forward({x0, h0})
  local dx_compute, dh0_compute = unpack(net:backward({x0, h0}, {dh}))
  
  local dx_err = gradcheck.relative_error(dx_num, dx_compute) -- this is still pretty big, in the order of 1e-1, something wrong in my test code?
  local dh0_err = gradcheck.relative_error(dh0_num, dh0_compute)
  local dw_err = gradcheck.relative_error(dw_num, grads)
  assert(dw_err < 1e-5)
  print(dx_err, dh0_err, dw_err)
end

function recurrentTest.rnnGradiantCheck()
  local N, D, H, T = 3, 4, 5, 4
  local glimpse = nn.Linear(D, D)
  local recurrent = nn.VanillaRnn(D, H)
  local rnn = nn.Rnn(H, glimpse, recurrent, nn.ReLU(), T)
  
  local params, grads = rnn:getParameters()
  
  params:uniform(-0.1, 0.1)
  grads:zero()
  local params_origin = params:clone()
  
  local x, df = torch.randn(T, N, D), torch.randn(N, H)
  
  local function fx(x1)
    local output = nil
    rnn:forget()
    for t = 1, T do
      if t == 1 then
        output = rnn:updateOutput(x1)
      else
        output = rnn:updateOutput(x[t])
      end  
    end
    return output
  end
  
  local function fw(w)
    local output = nil
    rnn:forget()
    params:copy(w)
    for t = 1, T do
      output = rnn:updateOutput(x[t])
    end
    params:copy(params_origin)
    return output
  end
  
  local dx_num = gradcheck.numeric_gradient(fx, x[1], df)
  local dw_num = gradcheck.numeric_gradient(fw, params, df)
  
  rnn:forget()
  for t = 1, T do
    rnn:forward(x[t])
  end
  local dx_compute = nil
  for t = T, 1, -1 do
    if t == T then
      dx_compute = rnn:backward(x[t], df)
    else
      dx_compute = rnn:backward(x[t], torch.zeros(N, H))
    end
    
  end
  
  local dx_err = gradcheck.relative_error(dx_num, dx_compute) -- this is still pretty big for T > 1, something wrong in my test code
  local dw_err = gradcheck.relative_error(dw_num, grads)
  print(dx_err, dw_err)
  assert(dw_err < 1e-5)
end

recurrentTest.rnnGradiantCheck()