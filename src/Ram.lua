require 'torch'
require 'nn'
require 'randomkit'

require 'LSTM'

local utils = require 'util.utils'
local glimpse = require 'Glimpse'

local ram, parent = torch.class('Ram', 'nn.Module')

local function clone_n_times(model, nClones, container)
  local t = {}
  for i = 1, nClones do
    local cloned = model:clone('weight', 'bias', 'gradWeight', 'gradBias')
    table.insert(t, cloned)
    container:add(cloned)
  end
  return t
end

function ram:__init(kwargs)
  -- D - input size, H - hidden size, T - depth of rnn, C - number of classes
  self.T = utils.get_kwarg(kwargs,'glimpses',4)
  self.D = utils.get_kwarg(kwargs,'glimpse_output_size', 256)
  self.H = utils.get_kwarg(kwargs, 'rnn_hidden_size', 128)
  self.C = utils.get_kwarg(kwargs, 'nClasses', 10)
  self.patchSize = utils.get_kwarg(kwargs,'patch_size', 8)
  self.location_gaussian_std = utils.get_kwarg(kwargs, 'location_gaussian_std', 0.1)
  self.exploration_rate = utils.get_kwarg(kwargs,'exploration_rate', 0.15)
  self.randomGlimpse = utils.get_kwarg(kwargs,'random_glimpse', false)
  
  local T, D, H, C, patchSize = self.T, self.D, self.H, self.C, self.patchSize
  local container = nn.Container()
  local glimpseNet = glimpse.createNet(patchSize, 
    utils.get_kwarg(kwargs,'glimpse_hidden_size',128), D, utils.get_kwarg(kwargs,'exploration_rate', 0.15))
  local glimpses = clone_n_times(glimpseNet, T, container)
  
  local rnn = nn.LSTM(D, H)
  local rnns = clone_n_times(rnn, T, container)
  
  local classificationNet = nn.Linear(H, C)
  container:add(classificationNet)
  
  local locationNet = nn.Sequential()
  locationNet:add(nn.Linear(H, 2))
--  locationNet:add(nn.Tanh())
  local locationNets = clone_n_times(locationNet, T, container)
  
  self.glimpses = glimpses
  self.rnns = rnns
  self.classificationNet = classificationNet
  self.locationNets = locationNets
  self.container = container

  self.rnnInputs = {}

  self.patch = torch.Tensor() -- N x T x (patchSize*patchSize)
--  self.l_m = torch.Tensor() -- N x T x 2
  self.l = torch.Tensor()   -- N x T x 2
  self.h = torch.Tensor()   -- N x T x H
  
  self.buffer1 = torch.Tensor()
  self.buffer2 = torch.Tensor()
  
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

function ram:forward(image, smapledLocation)
  local N, T, H, C, patchSize, std = image:size(1), self.T, self.H, self.C, self.patchSize, self.location_gaussian_std
  self.l:resize(N, T, 2)
  self.h:resize(N, T, H)
  self.patch:resize(N, T, patchSize*patchSize)
  local l, h, patch = self.l, self.h, self.patch
  
  local isGlimpseRandom = self.randomGlimpse
  local l_m = smapledLocation
  local computeLocation = not l_m
  
  -- initial location is the center of the image
  if computeLocation then
    l:zero()
    
    l_m = image.new():resize(N, T ,2):fill(0)
--    sampleFromGaussion(l_m[{{}, 1}], l[{{}, 1}], std)
  end

  patch[{{}, 1}] = glimpse.computePatch(image,l_m[{{}, 1}], patchSize, self.exploration_rate)
  local processedPatch = self.glimpses[1]:forward({patch[{{}, 1}], l_m[{{}, 1}]})
  local score = image.new():resize(N, C)
  local rnnInputs, rnnInput, prev_h, prev_c = {}, nil, nil, nil
  local buffer_lm = self.buffer2:resize(N, 2)
  for t = 1, T do
    if prev_h then
      rnnInput = {prev_c, prev_h, processedPatch}
    else
      rnnInput = processedPatch
    end
    table.insert(rnnInputs, rnnInput)
    h[{{}, t}], prev_c = self.rnns[t]:forward(rnnInput)
    
    if t < T then
      if not isGlimpseRandom then
        l[{{}, t+1}] = self.locationNets[t]:forward(h[{{}, t}])
        if computeLocation then
          sampleFromGaussion(l_m[{{}, t+1}], l[{{}, t+1}], std)
        end
      else
        l_m[{{}, t+1}]:uniform(-0.5, 0.5) 
      end
      
      patch[{{}, t+1}] = glimpse.computePatch(image,l_m[{{}, t+1}], patchSize, self.exploration_rate)
      processedPatch = self.glimpses[t+1]:forward({patch[{{}, t+1}], l_m[{{}, t+1}]})
    end
    prev_h = h[{{}, t}]
  end
  self.rnnInputs = rnnInputs
  local score = self.classificationNet:forward(prev_h)
  return score, l_m
end

function ram:backward(gradScore, l_m, reward)
  local N, T, l, sigma = gradScore:size(1), self.T, self.l, self.location_gaussian_std
  local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
  local reward_expand = nil
  if reward then
    reward_expand = self.buffer1:resize(N):copy(reward):view(N,1):expand(N,2)
  end
  local grad_h = self.classificationNet:backward(self.h[{{}, T}], gradScore)
  for t = T, 1, -1 do
    if rnn_grad_h_next then
      rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
    else
      rnnGradOutput = grad_h
    end
    rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[t]:backward(self.rnnInputs[t], rnnGradOutput))
    local grad_l = self.glimpses[t]:backward({self.patch[{{}, t}], l_m[{{}, t}]}, grad_g)[2]
    if t > 1 then
      if reward_expand then
        grad_l:cmul(reward_expand)
        grad_l:cmul(reward_expand):cmul(l_m[{{}, t}] - l[{{}, t}]):div(sigma * sigma)
        grad_h = self.locationNets[t-1]:backward(self.h[{{}, t-1}], grad_l)
      else
        grad_h:fill(0)
      end
      
    end
  end
end