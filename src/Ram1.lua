require 'torch'
require 'nn'
require 'randomkit'
require 'gnuplot'
require 'image'

require 'LSTM1'

local utils = require 'util.utils'
local glimpse = require 'Glimpse'

local ram, parent = torch.class('Ram1', 'nn.Module')

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
  self.T = utils.get_kwarg(kwargs,'glimpses',4)
  self.D = utils.get_kwarg(kwargs,'glimpse_output_size', 256)
  self.H = utils.get_kwarg(kwargs, 'rnn_hidden_size', 128)
  self.C = utils.get_kwarg(kwargs, 'nClasses', 10)
  self.patchSize = utils.get_kwarg(kwargs,'patch_size', 8)
  self.S = utils.get_kwarg(kwargs,'scales', 3)
  self.location_gaussian_std = utils.get_kwarg(kwargs, 'location_gaussian_std', 0.1)
  self.unitPixels = utils.get_kwarg(kwargs,'unitPixels', 15)
  self.randomGlimpse = utils.get_kwarg(kwargs,'random_glimpse', false)
  self.planRoute = utils.get_kwarg(kwargs,'plan_route', false)
  
  local T, D, H, C, patchSize, S = self.T, self.D, self.H, self.C, self.patchSize, self.S
  local container = nn.Container()
  local glimpseNet = glimpse.createNet1(patchSize, 
    utils.get_kwarg(kwargs,'glimpseHiddenSize',128), 
    utils.get_kwarg(kwargs,'locatorHiddenSize',128), 
    D, S)
  local glimpses = clone_n_times(container, glimpseNet, T)
  
  local rnn = nn.LSTM1(D, H)
  local rnns = clone_n_times(container, rnn, T)
  
  local classificationNet = nn.Linear(H, C)
--  local classificationNets = clone_n_times(container, classificationNet, E)
  
  local locationNet = nn.Sequential()
  locationNet:add(nn.Linear(H, 2))
--  locationNet:add(nn.Tanh())
  locationNet:add(nn.HardTanh())
  local locationNets = clone_n_times(container, locationNet, T)
  
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

function ram:forward(src)
  local N, T, H, C, patchSize, S, std = src:size(1), self.T, self.H, self.C, self.patchSize, self.S, self.location_gaussian_std
  local imageWidth = src:size(3)
  local isGlimpseRandom, planRoute = self.randomGlimpse, self.planRoute
  if not planRoute then
    self.l:resize(N, T, 2):zero()
    self.l_m:resize(N, T, 2):zero()
  end
  
  self.h:resize(N, T, H)
  self.patch:resize(N, T, S*patchSize*patchSize)
  local l, l_m, h, patch = self.l, self.l_m, self.h, self.patch
  self.initalHiddenStates = src.new():resize(N, H):zero()
  local initH = self.initalHiddenStates

  local rnnInputs, rnnInput, prev_h, prev_c = self.rnnInputs, nil, nil, nil
    
  for t = 1, T do
    if not planRoute then
      if t == 1 then
        l[{{}, t}] = self.locationNets[t]:forward(initH)
      else
        l[{{}, t}] = self.locationNets[t]:forward(h[{{}, t}])
      end
    
      l_m[{{}, t}]:normal():mul(std):add(l[{{}, t}])
      l_m[{{}, t}] = nn.HardTanh():cuda():forward(l_m[{{}, t}])
    end
      
    patch[{{}, t}] = glimpse.computePatch(src,l_m[{{}, t}], patchSize, self.unitPixels)
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
  local score = self.classificationNet:forward(prev_h)
  
  --TODO: todebug
--  plotPatch(T, src[1][1], patch[1][1], self.S)
  return score
end

function ram:backward(gradScore, reward)
  self.reward = reward
  local N, T, l, l_m, sigma = gradScore:size(1), self.T, self.l, self.l_m, self.location_gaussian_std
  
  local learnRoute = (not self.randomGlimpse) and (not self.planRoute)
  local reward_expand = nil
  if learnRoute then
    reward_expand = reward.new():resize(N):copy(reward):view(N, 1):expand(N, 2)
  end
  local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
  local grad_h = self.classificationNet:backward(self.h[{{}, T}], gradScore)
  for t = T, 1, -1 do
    if rnn_grad_h_next then
      rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
    else
      rnnGradOutput = grad_h
    end
    rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[t]:backward(self.rnnInputs[t], rnnGradOutput))
    self.glimpses[t]:backward({self.patch[{{}, t}], l_m[{{}, t}]}, grad_g)
      
    if learnRoute then
      local grad_l = (l_m[{{}, t}] - l[{{}, t}]) / (-sigma * sigma)
      grad_l:cmul(reward_expand)
      if t == 1 then
        self.locationNets[t]:backward(self.initalHiddenStates, grad_l)
      else
        grad_h = self.locationNets[t]:backward(self.h[{{}, t-1}], grad_l)  
      end
    else
      grad_h:fill(0)
    end
  end
end