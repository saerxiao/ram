-- Use the locator module from ER-ram
require 'torch'
require 'nn'
require 'randomkit'
require 'gnuplot'
require 'image'

require 'LSTM1'

-- this is used with train3

local utils = require 'util.utils'
local glimpse = require 'Glimpse'

local ram, parent = torch.class('Ram3', 'nn.Module')

local function clone_n_times(container, model, M, N)
  local t = {}
  for i = 1, M do
    if N then
      local tm = clone_n_times(container, model, N)
      table.insert(t, tm)
    else
      local cloned = model:clone('weight', 'bias', 'gradWeight', 'gradBias')
      table.insert(t, cloned)
      container:add(cloned)
    end
  end
  return t
end

function ram:__init(kwargs)
  -- D - input size, H - hidden size, T - depth of rnn, C - number of classes
  self.T = utils.get_kwarg(kwargs,'rho',4)
  self.D = utils.get_kwarg(kwargs,'imageHiddenSize', 256)
  self.H = utils.get_kwarg(kwargs, 'hiddenSize', 128)
  self.C = utils.get_kwarg(kwargs, 'nClasses', 10)
  self.patchSize = utils.get_kwarg(kwargs,'glimpsePatchSize', 8)
  self.S = utils.get_kwarg(kwargs,'glimpseDepth', 3)
  self.location_gaussian_std = utils.get_kwarg(kwargs, 'locatorStd', 0.1)
  self.unitPixels = utils.get_kwarg(kwargs,'unitPixels', 13)
  self.randomGlimpse = utils.get_kwarg(kwargs,'random_glimpse', false)
  self.planRoute = utils.get_kwarg(kwargs,'plan_route', false)
  local stochastic = utils.get_kwarg(kwargs,'stochastic', false)
  
  local T, D, H, C, patchSize, S = self.T, self.D, self.H, self.C, self.patchSize, self.S
  local container = nn.Container()
--  local glimpseNet = glimpse.createNet(patchSize, 
--    utils.get_kwarg(kwargs,'glimpseHiddenSize',128), D, S)
  local glimpseNet = glimpse.createNet1(patchSize, 
    utils.get_kwarg(kwargs,'glimpseHiddenSize',128), utils.get_kwarg(kwargs,'locatorHiddenSize',128), D, S)
  local glimpses = clone_n_times(container, glimpseNet, T)
  
  local rnn = nn.LSTM1(D, H)
  local rnns = clone_n_times(container, rnn, T)
  
  local classificationNet = nn.Sequential()
--  local classificationNet = nn.Linear(H, C)
  classificationNet:add(nn.Linear(H,C))
  classificationNet:add(nn.LogSoftMax())
--  local classificationNets = clone_n_times(container, classificationNet)
  
--  local locationNet = nn.Sequential()
--  local linearLayer = nn.Linear(H, 2)
--  linearLayer:noBias()
--  locationNet:add(linearLayer)
----  locationNet:add(nn.Tanh())
--  locationNet:add(nn.HardTanh())
--  local locationNets = clone_n_times(container, locationNet, T)
  
  local locator = nn.Sequential()
  locator:add(nn.Linear(H, 2))
  locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
  locator:add(nn.ReinforceNormal(self.location_gaussian_std, stochastic)) -- sample from normal, uses REINFORCE learning rule
  assert(locator:get(3).stochastic == stochastic, "Please update the dpnn package : luarocks install dpnn")
  locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
  locator:add(nn.MulConstant(self.unitPixels*2/28))
  local locationNets = clone_n_times(container, locator, T)
  
  self.glimpses = glimpses
  self.rnns = rnns
  self.classificationNet = classificationNet
  self.locationNets = locationNets
  self.container = container

  self.rnnInputs = {}

  self.patch = torch.Tensor() -- N x T x (S*patchSize*patchSize)
  self.l_m = torch.Tensor() -- N x T x 2
  self.l = torch.Tensor()   -- N x T x 2
  self.h = torch.Tensor()   -- N x T x H
  
  self.buffer1 = torch.Tensor()
  self.buffer2 = torch.Tensor()
  
  self.train = true
end

function ram:parameters()
  return self.container:parameters()
end

local function sampleFromGaussion(l_m, l, std)
  local i = 0
  l_m:apply(function() 
    i = i + 1
    local mod = i%2
    local x1 = (mod == 0) and i/2 or i/2+1
    local x3 = (mod == 0) and 2 or mod
    local sampled = torch.normal(l[x1][x3], std)
    return sampled
  end)
end

function ram:isTrain()
  self.train = true
end

function ram:predict()
  self.train = false
end

local function plotPatch(T, src, patch, scale)
  gnuplot.raw('set multiplot layout ' .. scale*2 .. ', 3')
  gnuplot.imagesc(src)
  local patch_view = patch:view(T, scale, 8, 8)
  for t = 1, T do
    for s = 1, scale do
      gnuplot.imagesc(patch_view[t][s])
    end
  end
end

function ram:updateOutput(src)
  local N, T, H, C, patchSize, S, std = src:size(1), self.T, self.H, self.C, self.patchSize, self.S, self.location_gaussian_std
  local imageWidth = src:size(3)
  
  self.l:resize(N, T, 2):zero()
  self.l_m:resize(N, T, 2):zero()
  
  self.h:resize(N, T, H):zero()
  self.patch:resize(N, T, S*patchSize*patchSize)
  local l, l_m, h, patch = self.l, self.l_m, self.h, self.patch
  self.initalHiddenStates = src.new():resize(N, H):zero()
  local initH = self.initalHiddenStates
  
  local score = src.new():resize(N, C)
  local rnnInputs = {}
   
  local rnnInput, prev_h, prev_c = {}, nil, nil, nil
    
    for t = 1, T do
       if t == 1 then
          l_m[{{}, t}] = self.locationNets[t]:forward(initH)
        else
          l_m[{{}, t}] = self.locationNets[t]:forward(h[{{}, t}])
        end
      
      patch[{{}, t}] = glimpse.computePatchRam3(src,l_m[{{}, t}], patchSize, self.unitPixels)
      local processedPatch = self.glimpses[t]:forward({patch[{{}, t}], l_m[{{}, t}]})
      
      if prev_h then
        rnnInput = {prev_c, prev_h, processedPatch}
      else
        rnnInput = processedPatch
      end
      table.insert(rnnInputs, rnnInput)
      h[{{}, t}], prev_c = self.rnns[t]:forward(rnnInput)
      
      prev_h = h[{{}, t}]
    end
    
  --TODO: todebug
--  plotPatch(T, src[1][1], patch[1][1], self.S)
  
  self.rnnInputs = rnnInputs
  self.output = self.classificationNet:forward(prev_h)
--  print(l)
  return self.output
end

function ram:backward(input, gradOutput)
  local N, T, l_m, sigma, reward = gradOutput:size(1), self.T, self.l_m, self.location_gaussian_std, self.reward
  
--  local reward_expand = reward:view(N, 1):expand(N, 2)
  local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
    local grad_h = self.classificationNet:backward(self.h[{{}, T}], gradOutput)
    local grad_action = input.new():resize(N, 2):zero()
    for t = T, 1, -1 do
      if rnn_grad_h_next then
        rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
      else
        rnnGradOutput = grad_h
      end
      rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[t]:backward(self.rnnInputs[t], rnnGradOutput))
      
--      local grad_l = self.glimpses[ep][t]:backward({self.patch[{{}, ep, t}], l_m[{{}, ep, t}]}, grad_g)[2]
      self.glimpses[t]:backward({self.patch[{{}, t}], l_m[{{}, t}]}, grad_g)
      
      if t == 1 then
        self.locationNets[t]:backward(self.initalHiddenStates, grad_action)
      else
        grad_h = self.locationNets[t]:backward(self.h[{{}, t-1}], grad_action)  
      end
    end
    self.gradInput = input.new():resizeAs(input):zero()
end

function ram:updateGradInput(input, gradOutput)
  self:backward(input, gradOutput)
  return self.gradInput
end


function ram:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

-- reward is a constant vector of size N. N is the batch size
function ram:reinforce(reward)
--  print(reward)
  self.reward = reward
  for i, module in ipairs(self.locationNets) do
    module:reinforce(reward)
  end
end