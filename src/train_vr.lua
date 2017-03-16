package.path = package.path .. ";/home/saxiao/ram/src/?.lua"

require 'torch'
require 'Ram1_vr'
require 'nn'
require 'optim'
require 'pl'
require 'image'
require 'dp'
require 'rnn'
require 'Rnn'
require 'LSTM1'
require 'VanillaRnn'
require 'RA1'

require 'RecurrentAttentionModel'
require 'ReinforceNormal'

local glimpseUtil = require 'Glimpse'

cmd = torch.CmdLine()
cmd:option('--myModel', false, 'use my implementation')
cmd:option('--myRnn', true, 'use my implementation of the Recurrent module')
cmd:option('--myVR', true, 'use my implementation of variance reduction')

cmd:option('-source', 'mnist.t7', 'directory for source data')
cmd:option('-dataset', '32x32', 'specify the variation of the dataset') --32x32, offcenter_100x100 
cmd:option('-validate_split', 0.9, 'sequence length')
cmd:option('-train_max_load', -1, 'loading size')

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-glimpses', 4, 'number of glimpses')
cmd:option('-patch_size', 8)
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('-scales', 1)
cmd:option('-glimpse_hidden_size', 128)
cmd:option('-locator_hidden_size', 128)
cmd:option('-imageHiddenSize', 256)
cmd:option('--FastLSTM', false, 'use LSTM instead of linear layer')
cmd:option('-rnnHiddenSize', 256)
cmd:option('-nClasses', 10)
cmd:option('-location_gaussian_std', 0.11) -- 0.1
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')
cmd:option('-unitPixels', 15, 'the locator unit (1,1) maps to pixels (15,15)')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('-lamda', 1)
cmd:option('-random_glimpse', false, 'whether the glimpses are random')
cmd:option('-plan_route', false, 'Use a planned route')

-- training options
cmd:option('-batch_size', 20, 'batch size')
cmd:option('-nepochs', 800, 'number of epochs')
cmd:option('-learning_rate',1e-2,'learning rate')  -- 1e-3 works fine for plan_route = true, 1e-4 works fine too, trains a little bit slower, but more stable
cmd:option('-momentum', 0.9,'momentum')
cmd:option('-lr_decay_every', 5)   -- lr_decay_every=3, lr_decay_factor=0.9 seems to work well for train_max_load=10000 and batch_size=100
cmd:option('-lr_decay_factor', 0.7)
cmd:option('-grad_clip', 5)
cmd:option('-uniform', 0.1)
cmd:option('-init_from', '') -- checkpoint/mnist.t7/32x32/var_reduction/epoch50_i9000.t7

cmd:option('-checkpoint_every', 100)
cmd:option('-save_every', 1000)
cmd:option('checkpoint_dir', 'checkpoint', 'dir for saving checkpoints')

cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
opt = cmd:parse(arg)

local type = 'torch.DoubleTensor'

-- load lib for gpu
if opt.gpuid > -1 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

local Loader = require 'Loader'
local loader = Loader.create(opt)

local net, ram = nil, nil
local start_epoch, start_i = 0, 0
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  net = checkpoint.model
  if opt.reset_iterations == 0 then
    start_epoch = checkpoint.epoch
    start_i = checkpoint.i
  end
else
  net = nn.Sequential()
  if opt.myModel then
    ram = Ram1_vr(opt)
    net:add(ram)
  else
    -- glimpse network (rnn input layer)
--   local locationSensor = nn.Sequential()
--   locationSensor:add(nn.SelectTable(2))
--   locationSensor:add(nn.Linear(2, opt.locator_hidden_size))
--   locationSensor:add(nn[opt.transfer]())
--
--   local glimpseSensor = nn.Sequential()
--   glimpseSensor:add(nn.SpatialGlimpse(opt.patch_size, opt.glimpseDepth, opt.glimpseScale):float())
--   glimpseSensor:add(nn.Collapse(3))
--   glimpseSensor:add(nn.Linear((opt.patch_size^2)*opt.glimpseDepth, opt.glimpse_hidden_size))
--   glimpseSensor:add(nn[opt.transfer]())
--
--   local glimpse = nn.Sequential()
--   glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
--   glimpse:add(nn.JoinTable(1,1))
--   glimpse:add(nn.Linear(opt.glimpse_hidden_size+opt.locator_hidden_size, opt.imageHiddenSize))
--   glimpse:add(nn[opt.transfer]())
--   if not opt.myRnn then
--    glimpse:add(nn.Linear(opt.imageHiddenSize, opt.rnnHiddenSize))
--   end
   
   glimpse = glimpseUtil.createNet1(opt.patch_size, opt.glimpse_hidden_size, opt.locator_hidden_size,
    opt.imageHiddenSize, opt.glimpseDepth)

   -- rnn recurrent layer
   local recurrent = nil
   if opt.FastLSTM then
     if opt.myRnn then
       recurrent = nn.LSTM1(opt.imageHiddenSize, opt.rnnHiddenSize)
     else
       recurrent = nn.FastLSTM(opt.rnnHiddenSize, opt.rnnHiddenSize)
     end     
   else
     if opt.myRnn then
       recurrent = nn.VanillaRnn(opt.imageHiddenSize, opt.rnnHiddenSize)
     else
       recurrent = nn.Linear(opt.rnnHiddenSize, opt.rnnHiddenSize)
     end
   end


   -- recurrent neural network
   local rnn = nil
   if opt.myRnn then
     rnn = nn.Rnn(opt.rnnHiddenSize, glimpse, recurrent, nn[opt.transfer](), opt.glimpses)
   else
     rnn = nn.Recurrent(opt.rnnHiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)
   end

   -- actions (locator)
   local imageSize = 32
   local locator = nn.Sequential()
   locator:add(nn.Linear(opt.rnnHiddenSize, 2))
   locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
--   locator:add(nn.ReinforceNormal(2*opt.location_gaussian_std, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
--   assert(locator:get(3).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
   locator:add(nn.ReNormal(2*opt.location_gaussian_std))
   locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
   locator:add(nn.MulConstant(opt.unitPixels*2/imageSize))

--   ram = nn.RecurrentAttention(rnn, locator, opt.glimpses, {opt.rnnHiddenSize})
--    ram = nn.RA1(rnn, locator, opt.glimpses, {opt.rnnHiddenSize}, opt.myRnn)
    ram = nn.RA(rnn, locator, opt.glimpses, {opt.rnnHiddenSize, opt.patch_size}, opt.myRnn)
   
   -- model is a reinforcement learning agent
   net:add(ram)

   -- classifier :
   net:add(nn.SelectTable(-1))
   net:add(nn.Linear(opt.rnnHiddenSize, opt.nClasses))
   net:add(nn.LogSoftMax())
  end

  if not opt.myVR then
  -- add the baseline reward predictor
    local seq = nn.Sequential()
    seq:add(nn.Constant(1,1))
    seq:add(nn.Add(1))
    local concat = nn.ConcatTable():add(nn.Identity()):add(seq)
    local concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)
    net:add(concat2)
  end
  
  if opt.uniform > 0 then
    for k,param in ipairs(net:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
    end
--    if opt.myRnn then
--      for k,param in ipairs(seq:parameters()) do
--        param:uniform(-opt.uniform, opt.uniform)
--      end
--    else
--      for k,param in ipairs(net:parameters()) do
--        param:uniform(-opt.uniform, opt.uniform)
--      end
--    end
  end
end

  local criterion = nil
  if opt.myVR then
    criterion = nn.ClassNLLCriterion()
  else
    criterion = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.VRClassReward(net, opt.lamda), nil, nn.Convert())) -- REINFORCE 
  end


-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterion = criterion:cuda()
  type = net:type()
end

local params, grads = net:getParameters()

local trainIter = loader:iterator("train")

local train_accuracy, pos_reward, neg_reward, zero_reward = 0, 0, 0, 0
local train_loss_history = {}
local train_accuracy_history = {}
local pos_reward_history = {}
local neg_reward_history = {}
local zero_reward_history = {}
local reward_mean_history = {}
local reward_history = {}
local location_history = {}
local location_mean_history = {}

local last_train_data, last_train_label = nil, nil
local rewardRollingMean = 1 + torch.uniform(-0.1, 0.1)
local n = 0
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label = trainIter.next_batch()
  data = data:type(type)
  label = label:type(type)
  last_train_data, last_train_label = data, label
  local N, T, randomGlimpse = data:size(1), opt.glimpses, opt.random_glimpse
  
  if opt.plan_route then
    local l_m = ram.l_m
    local unitPixel = opt.unitPixels
    l_m:resize(N,T, 2):zero()
    l_m[{{}, 1, 1}]:fill(-4/unitPixel)
    l_m[{{}, 1, 2}]:fill(-4/unitPixel)
    l_m[{{}, 2, 1}]:fill(-4/unitPixel)
    l_m[{{}, 2, 2}]:fill(4/unitPixel)
    l_m[{{}, 3, 1}]:fill(4/unitPixel)
    l_m[{{}, 3, 2}]:fill(4/unitPixel)
    l_m[{{}, 4, 1}]:fill(4/unitPixel)
    l_m[{{}, 4, 2}]:fill(-4/unitPixel)
  end

  local outputTable = net:forward(data)
  local loss = criterion:forward(outputTable, label)
  table.insert(train_loss_history, loss)
  
  local classifierOutput = nil
  if opt.myVR then
    classifierOutput = outputTable
  else
    classifierOutput = outputTable[1]
  end
  local _, predict = classifierOutput:max(2)
  predict = predict:squeeze():type(type)
  
  local reward = torch.eq(predict, label):type(type)
  train_accuracy = reward:sum()/N
  table.insert(train_accuracy_history, train_accuracy)
  
--  if not rewardRollingMean then
--    rewardRollingMean = reward:sum() / N
--  end
--  
--  rewardRollingMean = (rewardRollingMean * n + reward:sum()) / (n + N)
--  n = n + N
  
  local gradLoss = criterion:backward(outputTable, label)
  if opt.myVR then
    reward = (reward - train_accuracy) * opt.lamda
    reward:div(N)
    ram:reinforce(reward)
  end
  net:backward(data, gradLoss)
  
  if opt.grad_clip > 0 then
    grads:clamp(-opt.grad_clip, opt.grad_clip)
  end
  
  return loss, grads
end

local validateIter = loader:iterator("validate")
local function calAccuracy(data, label)
  ram:predict()
  local outputTable = net:forward(data) -- N x C
  local classifierOutput = nil
  if opt.myVR then
    classifierOutput = outputTable
  else
    classifierOutput = outputTable[1]
  end
  local _, predict = classifierOutput:max(2)
  predict = predict:squeeze():type(type)
  local hits = torch.eq(label, predict):sum()
  local nTotal = label:size(1)
    
  local loss = criterion:forward(outputTable, label)

  return loss, hits/nTotal
end

--local function calValidateAccuracy()
--  local data, label = validateIter.next_batch()
--  data = data:type(type)
--  label = label:type(type)
--  return calAccuracy(data, label)
--end

local valIter = loader:iterator("validate")
local function calValidateAccuracy()
  local nTotal, hits, loss = 0, 0, 0
  repeat
    local data, label = valIter.next_batch()
    data = data:type(type)
    label = label:type(type)
    local outputTable = net:forward(data) -- N x C
    local classifierOutput = nil
    if opt.myVR then
      classifierOutput = outputTable
    else
      classifierOutput = outputTable[1]
    end
    local _, predict = classifierOutput:max(2)
    predict = predict:squeeze():type(label:type())
    hits = hits + torch.eq(label, predict):sum()
    nTotal = nTotal + label:size(1)
    
    loss = loss + criterion:forward(outputTable, label) * label:size(1)
  until nTotal > 100
  return loss/nTotal, hits/nTotal
end

--if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
--lfs.chdir(opt.checkpoint_dir)
--if not path.exists(opt.source) then lfs.mkdir(opt.source) end
--lfs.chdir(opt.source)
--if not path.exists(opt.dataset) then lfs.mkdir(opt.dataset) end
--lfs.chdir(opt.dataset)

local val_loss_history = {}
local val_accuracy_history = {}
local iterations = {}
local epochs = {}
local iter_per_epoch = loader.split_size['train']
local lr = opt.learning_rate
local optim_opt = {learningRate = lr, momentum = opt.momentum}
local num_iterations = iter_per_epoch * opt.nepochs
local checkpoint = {}
for i = 1, num_iterations do
  local epoch = math.ceil(i / iter_per_epoch)
  
--  local _, loss = optim.adam(feval, params, optim_opt)
  local _, loss = optim.sgd(feval, params, optim_opt)

  local check_every = opt.checkpoint_every
  if (i-1) % iter_per_epoch == 0 then
   
    checkpoint.iter = i
    checkpoint.epoch = epoch
    checkpoint.model = net
    checkpoint.opt = opt
    checkpoint.loss = loss[1]
--    checkpoint.train_loss_after = train_loss_after
    local val_loss, val_accuracy = calValidateAccuracy("validate")
    checkpoint.validate_loss = val_loss
    checkpoint.validate_accuracy = val_accuracy
    
--    checkpoint.train_loss_history = train_loss_history
--    checkpoint.train_accuracy_history = train_accuracy_history
--    checkpoint.val_loss_history = val_loss_history
--    checkpoint.val_accuracy_history = val_accuracy_history
--    checkpoint.reward_mean_history = reward_mean_history
--    checkpoint.reward_history = reward_history
--    checkpoint.location_history = location_history
--    checkpoint.location_mean_history = location_mean_history
    print("i = ", i, " epoch = ", epoch, "train_loss = ", checkpoint.loss, "train_accuracy = ", train_accuracy, "val_loss = ", val_loss, "val_accuracy = ", val_accuracy)
--    print("rmean = ", rewardRollingMean, " reward = ", reward, " n = ", n)
    table.insert(val_loss_history, val_loss)
    table.insert(val_accuracy_history, val_accuracy)
    table.insert(iterations, i)
    table.insert(epochs, epoch)
    local save_every = opt.save_every
--    local savefile = string.format('epoch%d.t7', checkpoint.epoch + start_epoch)
    local savefile = string.format('saved-model/epoch%d.t7', checkpoint.epoch)
    torch.save(savefile, checkpoint)
  end
  
--  if i % iter_per_epoch == 0 and epoch % opt.lr_decay_every == 0 then
--    -- Maybe decay learning rate
--    local old_lr = optim_opt.learningRate
--    optim_opt = {learningRate = old_lr * opt.lr_decay_factor}
--  end
  lr = lr * (1 - i / num_iterations)
--  if i % iter_per_epoch == 0 then
--    lr = lr * (1 - epoch / opt.nepochs)
--  end
end

local file = string.format('stats.t7', hfolder)
local stats = {}
stats.opt = opt
stats.iterations = iterations
stats.epochs = epochs
stats.train_loss_history = train_loss_history
stats.train_accuracy = train_accuracy_history
stats.pos_reward = pos_reward_history
stats.neg_reward = neg_reward_history
stats.val_loss_history = val_loss_history
stats.val_accuracy_history = val_accuracy_history
torch.save(file, stats)