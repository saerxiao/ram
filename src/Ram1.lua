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
  -- D - input size, H - hidden size, T - depth of rnn, C - number of classes
  self.T = utils.get_kwarg(kwargs,'glimpses',4)
  self.D = utils.get_kwarg(kwargs,'glimpse_output_size', 256)
  self.H = utils.get_kwarg(kwargs, 'rnn_hidden_size', 128)
  self.C = utils.get_kwarg(kwargs, 'nClasses', 10)
  self.E = utils.get_kwarg(kwargs, 'episodes', 5)
  self.patchSize = utils.get_kwarg(kwargs,'patch_size', 8)
  self.S = utils.get_kwarg(kwargs,'scales', 3)
  self.location_gaussian_std = utils.get_kwarg(kwargs, 'location_gaussian_std', 0.1)
--  self.exploration_rate = utils.get_kwarg(kwargs,'exploration_rate', 0.15)
  self.unitPixels = utils.get_kwarg(kwargs,'unitPixels', 15)
  self.randomGlimpse = utils.get_kwarg(kwargs,'random_glimpse', false)
  self.planRoute = utils.get_kwarg(kwargs,'plan_route', false)
  
  local T, D, H, C, E, patchSize, S = self.T, self.D, self.H, self.C, self.E, self.patchSize, self.S
  local container = nn.Container()
--  local glimpseNet = glimpse.createNet(patchSize, 
--    utils.get_kwarg(kwargs,'glimpse_hidden_size',128), D, S)
  local glimpseNet = glimpse.createNet1(patchSize, 
    utils.get_kwarg(kwargs,'glimpseHiddenSize',128), 
    utils.get_kwarg(kwargs,'locatorHiddenSize',128), 
    D, S)
  local glimpses = clone_n_times(container, glimpseNet, E, T)
  
  local rnn = nn.LSTM1(D, H)
  local rnns = clone_n_times(container, rnn, E, T)
  
  local classificationNet = nn.Linear(H, C)
  local classificationNets = clone_n_times(container, classificationNet, E)
  
  local locationNet = nn.Sequential()
  locationNet:add(nn.Linear(H, 2))
--  locationNet:add(nn.Tanh())
  locationNet:add(nn.HardTanh())
  local locationNets = clone_n_times(container, locationNet, E, T)
  
  self.glimpses = glimpses
  self.rnns = rnns
  self.classificationNets = classificationNets
  self.locationNets = locationNets
  self.container = container

  self.rnnInputs = {}

  self.patch = torch.Tensor() -- N x E x T x (S*patchSize*patchSize)
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
  local E = self.train and self.E or 1
  local isGlimpseRandom, planRoute = self.randomGlimpse, self.planRoute
  if not planRoute then
    self.l:resize(N, E, T, 2):zero()
    self.l_m:resize(N, E, T, 2):zero()
  end
  
  self.h:resize(N, E, T, H)
  self.patch:resize(N, E, T, S*patchSize*patchSize)
  local l, l_m, h, patch = self.l, self.l_m, self.h, self.patch
  self.initalHiddenStates = src.new():resize(N, H):zero()
  local initH = self.initalHiddenStates
--  l:zero()    
--  if planRoute then
--    local unitPixel = self.unitPixels
--    l_m[{{}, {}, 1, 1}]:fill(-4/unitPixel)
--    l_m[{{}, {}, 1, 2}]:fill(-4/unitPixel)
--    l_m[{{}, {}, 2, 1}]:fill(-4/unitPixel)
--    l_m[{{}, {}, 2, 2}]:fill(4/unitPixel)
--    l_m[{{}, {}, 3, 1}]:fill(4/unitPixel)
--    l_m[{{}, {}, 3, 2}]:fill(4/unitPixel)
--    l_m[{{}, {}, 4, 1}]:fill(4/unitPixel)
--    l_m[{{}, {}, 4, 2}]:fill(-4/unitPixel)
--  else
--    l_m:zero()
--  end
  
--    sampleFromGaussion(l_m[{{}, 1}], l[{{}, 1}], std)
  
  local score = src.new():resize(N, E, C)
  local rnnInputs = {}
  for ep = 1, E do
--    patch[{{}, ep, 1}] = glimpse.computePatch(src,l_m[{{}, ep, 1}], patchSize, self.unitPixels)
----    patch[{{}, ep, 1}] = glimpse.computePatchMultipleScale1(src,l_m[{{}, ep, 1}], patchSize, self.exploration_rate, self.S)
--    local processedPatch = self.glimpses[ep][1]:forward({patch[{{}, ep, 1}], l_m[{{}, ep, 1}]})
    
    local rnnInputsEp, rnnInput, prev_h, prev_c = {}, nil, nil, nil
    
    for t = 1, T do
      if not planRoute then
        if t == 1 then
          l[{{}, ep, t}] = self.locationNets[ep][t]:forward(initH)
        else
          l[{{}, ep, t}] = self.locationNets[ep][t]:forward(h[{{}, ep, t}])
        end
    
        l_m[{{}, ep, t}]:normal():mul(std):add(l[{{}, ep, t}])
        l_m[{{}, ep, t}] = nn.HardTanh():cuda():forward(l_m[{{}, ep, t}])
      end
      
      patch[{{}, ep, t}] = glimpse.computePatch(src,l_m[{{}, ep, t}], patchSize, self.unitPixels)
      local processedPatch = self.glimpses[ep][t]:forward({patch[{{}, ep, t}], l_m[{{}, ep, t}]})
      
      if prev_h then
        rnnInput = {prev_c, prev_h, processedPatch}
      else
        rnnInput = processedPatch
      end
      table.insert(rnnInputsEp, rnnInput)
      h[{{}, ep, t}], prev_c = self.rnns[ep][t]:forward(rnnInput)
      
--      if t < T then
--        if isGlimpseRandom then
----          l_m[{{}, ep, t+1}]:uniform(-0.5, 0.5)
--          l_m[{{}, ep, t+1}]:uniform(-1, 1) 
----          l_m[{{}, ep, t+1}]:zero() 
--        elseif not planRoute then
--          l[{{}, ep, t+1}] = self.locationNets[ep][t]:forward(h[{{}, ep, t}])
----          sampleFromGaussion(l_m[{{}, ep, t+1}], l[{{}, ep, t+1}], std)
----          l_m[{{}, ep, t+1}] = nn.HardTanh():cuda():forward(l_m[{{}, ep, t+1}])
--          l_m[{{}, ep, t}]:normal():mul(std):add(l[{{}, ep, t}])
--          l_m[{{}, ep, t}] = nn.HardTanh():cuda():forward(l_m[{{}, ep, t}])
----          l_m[{{}, ep, t+1}]:mul(self.unitPixels*2/imageWidth)
--        end
--      
--        patch[{{}, ep, t+1}] = glimpse.computePatch(src,l_m[{{}, ep, t+1}], patchSize, self.unitPixels)
----        patch[{{}, ep, t+1}] = glimpse.computePatchMultipleScale1(src,l_m[{{}, ep, t+1}], patchSize, self.exploration_rate, self.S)
--        processedPatch = self.glimpses[ep][t+1]:forward({patch[{{}, ep, t+1}], l_m[{{}, ep, t+1}]})
--      end
      prev_h = h[{{}, ep, t}]
    end
    score[{{}, ep}] = self.classificationNets[ep]:forward(prev_h)
    table.insert(rnnInputs, rnnInputsEp)

  end
  
  --TODO: todebug
--  plotPatch(T, src[1][1], patch[1][1], self.S)
  
  self.rnnInputs = rnnInputs 
  return score
end

function ram:backward(gradScore, reward)
  self.reward = reward
  local N, E, T, l, l_m, sigma = gradScore:size(1), self.E, self.T, self.l, self.l_m, self.location_gaussian_std
  
  local learnRoute = (not self.randomGlimpse) and (not self.planRoute)
  local reward_expand = nil
  if learnRoute then
    reward_expand = reward.new():resize(N, E):copy(reward):view(N,E,1):expand(N,E,2)
--    reward_expand:add(-1, l)
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
--      local grad_l = self.glimpses[ep][t]:backward({self.patch[{{}, ep, t}], l_m[{{}, ep, t}]}, grad_g)[2]
      self.glimpses[ep][t]:backward({self.patch[{{}, ep, t}], l_m[{{}, ep, t}]}, grad_g)
      
      if learnRoute then
        local grad_l = (l_m[{{}, ep, t}] - l[{{}, ep, t}]) / (-sigma * sigma)
        grad_l:cmul(reward_expand)
        if t == 1 then
          self.locationNets[ep][t]:backward(self.initalHiddenStates, grad_l)
        else
          grad_h = self.locationNets[ep][t]:backward(self.h[{{}, ep, t-1}], grad_l)  
        end
      else
        grad_h:fill(0)
      end
      
--      if t > 1 then
--        if learnRoute then
----          local grad_l = reward.new():resize(N,2):copy(reward_expand[{{}, ep}])
------          grad_l:cmul(reward_expand[{{}, ep}])
----          grad_l:cmul(l_m[{{}, ep, t}] - l[{{}, ep, t}]):div(-sigma * sigma)
--          local grad_l = (l_m[{{}, ep, t}] - l[{{}, ep, t}]) / (-sigma * sigma)
--          grad_l:cmul(reward_expand)
--          grad_h = self.locationNets[ep][t-1]:backward(self.h[{{}, ep, t-1}], grad_l)  
--        else
--          grad_h:fill(0)
--        end
--      end
    end
  end
end