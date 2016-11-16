require 'torch'
require 'nn'
require 'randomkit'

require 'LSTM'

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
  -- D - input size, H - hidden size, T - depth of rnn, C - number of classes
  self.T = utils.get_kwarg(kwargs,'glimpses',4)
  self.D = utils.get_kwarg(kwargs,'glimpse_output_size', 256)
  self.H = utils.get_kwarg(kwargs, 'rnn_hidden_size', 128)
  self.C = utils.get_kwarg(kwargs, 'nClasses', 10)
  self.E = utils.get_kwarg(kwargs, 'episodes', 5)
  self.patchSize = utils.get_kwarg(kwargs,'patch_size', 8)
  self.location_gaussian_std = utils.get_kwarg(kwargs, 'location_gaussian_std', 0.1)
  self.exploration_rate = utils.get_kwarg(kwargs,'exploration_rate', 0.15)
  self.randomGlimpse = utils.get_kwarg(kwargs,'random_glimpse', false)
  
  local T, D, H, C, E, patchSize = self.T, self.D, self.H, self.C, self.E, self.patchSize
  local container = nn.Container()
  local glimpseNet = glimpse.createNet(patchSize, 
    utils.get_kwarg(kwargs,'glimpse_hidden_size',128), D, utils.get_kwarg(kwargs,'exploration_rate', 0.15))
  local glimpses = clone_n_times(container, glimpseNet, E, T)
  
  local rnn = nn.LSTM(D, H)
  local rnns = clone_n_times(container, rnn, E, T)
  
  local classificationNet = nn.Linear(H, C)
  local classificationNets = clone_n_times(container, classificationNet, E)
  
  local locationNet = nn.Sequential()
  locationNet:add(nn.Linear(H, 2))
--  locationNet:add(nn.Tanh())
  local locationNets = clone_n_times(container, locationNet, E, T)
  
  self.glimpses = glimpses
  self.rnns = rnns
  self.classificationNets = classificationNets
  self.locationNets = locationNets
  self.container = container

  self.rnnInputs = {}

  self.patch = torch.Tensor() -- N x E x T x (patchSize*patchSize)
  self.l_m = torch.Tensor() -- N x E x T x 2
  self.l = torch.Tensor()   -- N x E x T x 2
  self.h = torch.Tensor()   -- N x E x T x H
  
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

function ram:forward(image)
  local N, T, H, C, patchSize, std = image:size(1), self.T, self.H, self.C, self.patchSize, self.location_gaussian_std
  local E = self.train and self.E or 1
  self.l:resize(N, E, T, 2)
  self.l_m:resize(N, E, T, 2)
  self.h:resize(N, E, T, H)
  self.patch:resize(N, E, T, patchSize*patchSize)
  local l, l_m, h, patch = self.l, self.l_m, self.h, self.patch
  
  local isGlimpseRandom = self.randomGlimpse
  l:zero()    
  l_m:zero()
--    sampleFromGaussion(l_m[{{}, 1}], l[{{}, 1}], std)
  
  local score = image.new():resize(N, E, C)
  local rnnInputs = {}
  for ep = 1, E do
    patch[{{}, ep, 1}] = glimpse.computePatch(image,l_m[{{}, ep, 1}], patchSize, self.exploration_rate)
    local processedPatch = self.glimpses[ep][1]:forward({patch[{{}, ep, 1}], l_m[{{}, ep, 1}]})
    
    local rnnInputsEp, rnnInput, prev_h, prev_c = {}, nil, nil, nil
    
    for t = 1, T do
      if prev_h then
        rnnInput = {prev_c, prev_h, processedPatch}
      else
        rnnInput = processedPatch
      end
      table.insert(rnnInputsEp, rnnInput)
      h[{{}, ep, t}], prev_c = self.rnns[ep][t]:forward(rnnInput)
      
      if t < T then
        if isGlimpseRandom then
          l_m[{{}, ep, t+1}]:uniform(-0.5, 0.5) 
        else
          l[{{}, ep, t+1}] = self.locationNets[ep][t]:forward(h[{{}, ep, t}])
          sampleFromGaussion(l_m[{{}, ep, t+1}], l[{{}, ep, t+1}], std)
        end
      
        patch[{{}, ep, t+1}] = glimpse.computePatch(image,l_m[{{}, ep, t+1}], patchSize, self.exploration_rate)
        processedPatch = self.glimpses[ep][t+1]:forward({patch[{{}, ep, t+1}], l_m[{{}, ep, t+1}]})
      end
      prev_h = h[{{}, ep, t}]
    end
    score[{{}, ep}] = self.classificationNets[ep]:forward(prev_h)
    table.insert(rnnInputs, rnnInputsEp)
  end
  
  self.rnnInputs = rnnInputs 
  return score
end

function ram:backward(gradScore, reward)
  local N, E, T, l, l_m, sigma = gradScore:size(1), self.E, self.T, self.l, self.l_m, self.location_gaussian_std
  
  local isGlimpseRandom = self.randomGlimpse
  local reward_expand = nil
  if not isGlimpseRandom then
    reward_expand = reward.new():resize(N, E):copy(reward):view(N,E,1):expand(N,E,2)
  end
  for ep = 1, E do
    local rnnGradOutput, rnn_grad_c_next, rnn_grad_h_next, grad_g = nil, nil, nil, nil
    local grad_h = self.classificationNets[ep]:backward(self.h[{{}, ep, T}], gradScore[{{}, ep}])
    for t = T, 1, -1 do
      if rnn_grad_h_next then
        rnnGradOutput = {rnn_grad_c_next, rnn_grad_h_next, grad_h}
      else
        rnnGradOutput = grad_h
      end
      rnn_grad_c_next, rnn_grad_h_next, grad_g = unpack(self.rnns[ep][t]:backward(self.rnnInputs[ep][t], rnnGradOutput))
      local grad_l = self.glimpses[ep][t]:backward({self.patch[{{}, ep, t}], l_m[{{}, ep, t}]}, grad_g)[2]
      if t > 1 then
        if isGlimpseRandom then
          grad_h:fill(0)
        else
--          grad_l:cmul(reward_expand[{{}, ep}])
--          grad_l:cmul(reward_expand[{{}, ep}]):cmul(l_m[{{}, ep, t}] - l[{{}, ep, t}]):div(sigma * sigma)
          grad_h = self.locationNets[ep][t-1]:backward(self.h[{{}, ep, t-1}], grad_l)  
        end
      end
    end
  end
end