-- use the glimpse network from ER-rnn

require 'torch'
require 'nn'
require 'randomkit'
require 'gnuplot'
require 'image'
--require 'dp'
--require 'rnn'

require 'LSTM1'

local utils = require 'util.utils'
local glimpse = require 'Glimpse'

local ram, parent = torch.class('Ram4', 'nn.Module')

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
  
  local T, D, H, C, patchSize, S = self.T, self.D, self.H, self.C, self.patchSize, self.S
  local container = nn.Container()

--  local glimpseNet = glimpse.createNet1(patchSize, 
--    utils.get_kwarg(kwargs,'glimpseHiddenSize',128), utils.get_kwarg(kwargs,'locatorHiddenSize',128), D, S)
  
  local glimpseHiddenSize = utils.get_kwarg(kwargs,'glimpseHiddenSize',128)
  local locatorHiddenSize = utils.get_kwarg(kwargs,'locatorHiddenSize',128)
  local locationSensor = nn.Sequential()
  locationSensor:add(nn.SelectTable(2))
  locationSensor:add(nn.Linear(2,locatorHiddenSize))
  locationSensor:add(nn.ReLU())

  local glimpseSensor = nn.Sequential()
  glimpseSensor:add(nn.SpatialGlimpse(patchSize, self.S, utils.get_kwarg(kwargs,'glimpseScale', 2)):float())
  glimpseSensor:add(nn.Collapse(3))
  glimpseSensor:add(nn.Linear((patchSize^2)*self.S, glimpseHiddenSize))
  glimpseSensor:add(nn.ReLU())

  local glimpseNet = nn.Sequential()
  glimpseNet:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
  glimpseNet:add(nn.JoinTable(1,1))
  glimpseNet:add(nn.Linear(glimpseHiddenSize+locatorHiddenSize, self.D))
  glimpseNet:add(nn.ReLU())
--  glimpseNet:add(nn.Linear(D, H))
  
  local glimpses = clone_n_times(container, glimpseNet, T)
  
  local rnn = nn.LSTM1(D, H)
  local rnns = clone_n_times(container, rnn, T)
  
  local classificationNet = nn.Sequential()
  classificationNet:add(nn.Linear(H,C))
  classificationNet:add(nn.LogSoftMax())

  local locationNet = nn.Sequential()
  local linearLayer = nn.Linear(H, 2)
  locationNet:add(linearLayer)
  locationNet:add(nn.HardTanh())
  local locationNets = clone_n_times(container, locationNet, T)
  
  self.glimpses = glimpses
  self.rnns = rnns
  self.classificationNet = classificationNet
  self.locationNets = locationNets
  self.container = container

  self.rnnInputs = {}

  self.patch = torch.Tensor() -- N x T x (S*patchSize*patchSize)
  self.g = torch.Tensor() -- input to RNN, N x T x D
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
  local N, T, D, H, C, patchSize, S, std = src:size(1), self.T, self.D, self.H, self.C, self.patchSize, self.S, self.location_gaussian_std
  local imageWidth = src:size(3)
  self.l:resize(N, T, 2)
  self.l_m:resize(N, T, 2)
  local isGlimpseRandom, planRoute = self.randomGlimpse, self.planRoute
  if not planRoute then
    self.l:resize(N, T, 2):zero()
    self.l_m:resize(N, T, 2):zero()
  end
  self.g:resize(N, T, D)
  self.h:resize(N, T, H)
  self.patch:resize(N, T, S*patchSize*patchSize)
  local l, l_m, g, h, patch = self.l, self.l_m, self.g, self.h, self.patch
  self.initalHiddenStates = src.new():resize(N, H):zero()
  local initH = self.initalHiddenStates
  if planRoute then
    local unitPixel = self.unitPixels
    l_m[{{}, 1, 1}]:fill(-4/unitPixel)
    l_m[{{}, 1, 2}]:fill(-4/unitPixel)
    l_m[{{}, 2, 1}]:fill(-4/unitPixel)
    l_m[{{}, 2, 2}]:fill(4/unitPixel)
    l_m[{{}, 3, 1}]:fill(4/unitPixel)
    l_m[{{}, 3, 2}]:fill(4/unitPixel)
    l_m[{{}, 4, 1}]:fill(4/unitPixel)
    l_m[{{}, 4, 2}]:fill(-4/unitPixel)
  end
  
--    sampleFromGaussion(l_m[{{}, 1}], l[{{}, 1}], std)
  
  local score = src.new():resize(N, C)
  local rnnInputs = {}
--  patch[{{}, 1}] = glimpse.computePatch(src,l_m[{{}, 1}], patchSize, self.unitPixels)
----    patch[{{}, ep, 1}] = glimpse.computePatchMultipleScale1(src,l_m[{{}, ep, 1}], patchSize, self.exploration_rate, self.S)
--    local processedPatch = self.glimpses[1]:forward({patch[{{}, 1}], l_m[{{}, 1}]})
    
    local rnnInput, prev_h, prev_c = {}, nil, nil, nil
    
    for t = 1, T do
      if not planRoute then
        if t == 1 then
          l[{{}, t}] = self.locationNets[t]:forward(initH)
        else
          l[{{}, t}] = self.locationNets[t]:forward(h[{{}, t}])
        end
        l_m[{{}, t}]:normal():mul(std):add(l[{{}, t}])
        l_m[{{}, t}] = nn.HardTanh():cuda():forward(l_m[{{}, t}])
--      l_m[{{}, t}]:mul(self.unitPixels*2/imageWidth)
      end
      
--      patch[{{}, t}] = glimpse.computePatch(src,l_m[{{}, t}], patchSize, self.unitPixels)
--      local processedPatch = self.glimpses[t]:forward({patch[{{}, t}], l_m[{{}, t}]})
      g[{{}, t}] = self.glimpses[t]:forward({src, l_m[{{}, t}]})
      
      if prev_h then
        rnnInput = {prev_c, prev_h, g[{{}, t}]}
      else
        rnnInput = g[{{}, t}]
      end
      table.insert(rnnInputs, rnnInput)
      h[{{}, t}], prev_c = self.rnns[t]:forward(rnnInput)
      
--      if t < T then
--        if isGlimpseRandom then
----          l_m[{{}, ep, t+1}]:uniform(-0.5, 0.5)
--          l_m[{{}, t+1}]:uniform(-1, 1) 
----          l_m[{{}, ep, t+1}]:zero() 
--        elseif not planRoute then
--          l[{{}, t+1}] = self.locationNets[t]:forward(h[{{}, t}])
--          l_m[{{}, t+1}]:normal():mul(std):add(l[{{}, t+1}])
----          sampleFromGaussion(l_m[{{}, t+1}], l[{{}, t+1}], std)
----          l_m[{{}, t+1}]:tanh()
--          l_m[{{}, t+1}] = nn.HardTanh():cuda():forward(l_m[{{}, t+1}])
--          l_m[{{}, t+1}]:mul(self.unitPixels*2/imageWidth)
--        end
--      
--        patch[{{}, t+1}] = glimpse.computePatch(src,l_m[{{}, t+1}], patchSize, self.unitPixels)
----        patch[{{}, ep, t+1}] = glimpse.computePatchMultipleScale1(src,l_m[{{}, ep, t+1}], patchSize, self.exploration_rate, self.S)
--        processedPatch = self.glimpses[t+1]:forward({patch[{{}, t+1}], l_m[{{}, t+1}]})
--      end
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
  local N, T, l, l_m, sigma, reward = gradOutput:size(1), self.T, self.l, self.l_m, self.location_gaussian_std, self.reward
  
  local learnRoute = (not self.randomGlimpse) and (not self.planRoute)
  local reward_expand = nil
  if learnRoute then
    reward_expand = reward:view(N, 1):expand(N, 2)
  end

--  local reward_expand = reward:view(N, 1):expand(N, 2)
  local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
    local grad_h = self.classificationNet:backward(self.h[{{}, T}], gradOutput)
    for t = T, 1, -1 do
      if rnn_grad_h_next then
        rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
      else
        rnnGradOutput = grad_h
      end
      rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[t]:backward(self.rnnInputs[t], rnnGradOutput))
      
--      self.glimpses[t]:backward({self.patch[{{}, t}], l_m[{{}, t}]}, grad_g)
      self.glimpses[t]:backward({input, l_m[{{}, t}]}, grad_g)
      if learnRoute then
        local grad_l = (l_m[{{}, t}] - l[{{}, t}]) / (-sigma * sigma)
        grad_l:cmul(reward_expand)
        if t == 1 then
          self.locationNets[t]:backward(self.initalHiddenStates, grad_l)
        else
        --          grad_l:cmul(reward_expand[{{}, ep}])
--        local grad_l = reward.new():resize(N):copy(reward):view(N,1):expand(N,2)
--        grad_l:cmul(l_m[{{}, t}] - l[{{}, t}]):div(-sigma * sigma)
        
          grad_h = self.locationNets[t]:backward(self.h[{{}, t-1}], grad_l)  
        end
      else
        grad_h:fill(0)
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
end