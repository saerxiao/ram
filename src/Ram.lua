require 'torch'
require 'nn'
require 'randomkit'

require 'LSTM'

local utils = require 'util.utils'
local glimpse = require 'Glimpse'

local ram, parent = torch.class('Ram', 'nn.Module')

local function clone_n_times(model, nClones, container)
  local t = {}
  table.insert(t, model)
  for i = 2, nClones do
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
  
  local T, D, H, C, patchSize = self.T, self.D, self.H, self.C, self.patchSize
  local container = nn.Container()
  local glimpseNet = glimpse.createNet(patchSize, 
    utils.get_kwarg(kwargs,'glimpse_hidden_size',128), D, utils.get_kwarg(kwargs,'exploration_rate', 0.15))
  local glimpses = clone_n_times(glimpseNet, T, container)
  
  local rnn = nn.LSTM(D, H)
  local rnns = clone_n_times(rnn, T, container)
  
  local classificationNet = nn.Sequential()
  classificationNet:add(nn.Linear(H, C))
  classificationNet:add(nn.Tanh())
  local classificationNets = clone_n_times(classificationNet, T, container)
  
  local locationNet = nn.Linear(H, 2)
  local locationNets = clone_n_times(locationNet, T, container)
  
  self.glimpses = glimpses
  self.rnns = rnns
  self.classificationNets = classificationNets
  self.classificationCriterior = nn.CrossEntropyCriterion()
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

function ram:ship2cuda()
  self:cuda()
  self.classificationCriterior:cuda()
  return self
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
--  self.l_m:resize(N, T, 2)
  self.l:resize(N, T, 2)
  self.h:resize(N, T, H)
  self.patch:resize(N, T, patchSize*patchSize)
  local l, h, patch = self.l, self.h, self.patch
  
  local l_m = smapledLocation
  local computeLocation = not l_m
  
  -- initial location is the center of the image
--  local l_m0 = self.buffer1:resize(N, 2):fill(0)
  if computeLocation then
    l[{{}, 1}]:fill(0)  -- dummy
    
    l_m = image.new():resize(N, T ,2)
--    sampleFromGaussion(l_m[{{}, 1}], l[{{}, 1}], std)
    l_m[{{}, 1}]:fill(0)
  end
--  local l_m, l, h, patch = self.l_m, self.l, self.h, self.patch
  
--  l_m[{{}, 1}]:uniform(-0.5, 0.5)
--  l_m[{{}, 1}]:fill(0)
  patch[{{}, 1}] = glimpse.computePatch(image,l_m[{{}, 1}], patchSize, self.exploration_rate)
  local processedPatch = self.glimpses[1]:forward({patch[{{}, 1}], l_m[{{}, 1}]})
  local score = image.new():resize(N, T, C)
  local rnnInputs, rnnInput, prev_h, prev_c = {}, nil, nil, nil
--  local std = self.buffer1:resize(N, 2):fill(self.location_gaussian_std)
  local buffer_lm = self.buffer2:resize(N, 2)
  for t = 1, T do
    if prev_h then
      rnnInput = {prev_c, prev_h, processedPatch}
    else
      rnnInput = processedPatch
    end
    table.insert(rnnInputs, rnnInput)
    h[{{}, t}], prev_c = self.rnns[t]:forward(rnnInput)
    score[{{}, t}] = self.classificationNets[t]:forward(h[{{}, t}])
    
    if t < T then
      l[{{}, t+1}] = self.locationNets[t]:forward(h[{{}, t}])
      if computeLocation then
--      randomkit.normal(buffer_lm, l[{{}, t}], std)
--      l_m[{{}, t+1}]:copy(buffer_lm)
        sampleFromGaussion(l_m[{{}, t+1}], l[{{}, t+1}], std)
      end
      patch[{{}, t+1}] = glimpse.computePatch(image,l_m[{{}, t+1}], patchSize, self.exploration_rate)
      processedPatch = self.glimpses[t+1]:forward({patch[{{}, t+1}], l_m[{{}, t+1}]})
    end
    prev_h = h[{{}, t}]
--    prev_c = cell
  end
  self.rnnInputs = rnnInputs
  return score, l_m
end

function ram:computeLoss(score, label)
  return self.classificationCriterior:forward(score[{{}, self.T}], label)
end

-- for each sample, reward and b are scaler, so in batch, reward and b are vector of size N, 
function ram:backwardFromClassification(classScore, label, l_m)
  local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
  for t = self.T, 1, -1 do
    local gradClassScore = self.classificationCriterior:backward(classScore[{{}, t}], label)
    local grad_h = self.classificationNets[t]:backward(self.h[{{}, t}], gradClassScore)
    if not rnn_grad_h_next then
      rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
    else
      rnnGradOutput = grad_h
    end
    rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[t]:backward(self.rnnInputs[t], rnnGradOutput))
    self.glimpses[t]:backward({self.patch[{{}, t}], l_m[{{}, t}]}, grad_g)
  end
end

-- reward - vector of N
function ram:backwardFromLocation(reward, l_m)
  local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
  local N, sigma, l = reward:size(1), self.location_gaussian_std, self.l
  local reward_expand = self.buffer1:resize(N):copy(reward):view(N,1):expand(N,2)
  for t = self.T, 2, -1 do
    local grad_l = torch.cmul(l_m[{{}, t}] - l[{{}, t}], reward_expand)/(sigma * sigma)
    local grad_h = self.locationNets[t]:backward(self.h[{{}, t-1}], grad_l)
    if not rnn_grad_h_next then
      rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
    else
      rnnGradOutput = grad_h
    end
    rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[t-1]:backward(self.rnnInputs[t-1], rnnGradOutput))
    self.glimpses[t]:backward({self.patch[{{}, t}], l_m[{{}, t}]}, grad_g)
  end
end