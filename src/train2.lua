package.path = package.path .. ";/home/saxiao/ram/src/?.lua"

require 'torch'
require 'nn'
require 'optim'
require 'pl'
require 'image'
require 'Ram1'

cmd = torch.CmdLine()
cmd:option('-source', 'mnist.t7', 'directory for source data')
cmd:option('-dataset', '32x32', 'specify the variation of the dataset') --32x32, offcenter_100x100 
cmd:option('-validate_split', 0.9, 'sequence length')
cmd:option('-train_max_load', 10000, 'loading size')

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-glimpses', 4, 'number of glimpses')
cmd:option('-patch_size', 8)
cmd:option('-scales', 1)
cmd:option('-glimpse_hidden_size', 128)
cmd:option('-locator_hidden_size', 128)
cmd:option('-glimpse_output_size', 256)
cmd:option('-rnn_hidden_size', 256)
cmd:option('-nClasses', 10)
cmd:option('-location_gaussian_std', 0.22) -- 0.1
--cmd:option('-exploration_rate', 1/32)
cmd:option('-unitPixels', 15, 'the locator unit (1,1) maps to pixels (15,15)')
cmd:option('-episodes', 1, 'number of episodes')  -- 10 not better
cmd:option('-lamda', 1)
cmd:option('-random_glimpse', false, 'whether the glimpses are random')
cmd:option('-plan_route', true, 'Use a planned route')

-- training options
cmd:option('-batch_size', 20, 'batch size')
cmd:option('-nepochs', 800, 'number of epochs')
cmd:option('-learning_rate',1e-4,'learning rate')  -- 1e-3 works fine for plan_route = true, 1e-4 works fine too, trains a little bit slower, but more stable
cmd:option('-momentum', 0.9,'momentum')
cmd:option('-lr_decay_every', 5)   -- lr_decay_every=3, lr_decay_factor=0.9 seems to work well for train_max_load=10000 and batch_size=100
cmd:option('-lr_decay_factor', 0.7)
cmd:option('-grad_clip', 5)
cmd:option('-init_from', '') -- checkpoint/mnist.t7/32x32/var_reduction/epoch50_i9000.t7

cmd:option('-checkpoint_every', 100)
cmd:option('-save_every', 1000)
cmd:option('checkpoint_dir', 'checkpoint', 'dir for saving checkpoints')
--cmd:option('checkpoint_subdir', 'ram_nmist', 'dir for saving checkpoints')

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

local net = nil
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
  net = Ram1(opt)
end

local criterior = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterior = criterior:cuda()
  type = net:type()
end

local params, grads = net:getParameters()

local trainIter = loader:iterator("train")

local train_loss, train_accuracy, pos_reward, neg_reward, zero_reward = 0, 0, 0, 0, 0
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
local rewardRollingMean = 0
local n = 0
local feval = function(w)
  net:isTrain()
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label = trainIter.next_batch()
  data = data:type(type)
  label = label:type(type)
  last_train_data, last_train_label = data, label
  local N, E, T, randomGlimpse = data:size(1), opt.episodes, opt.glimpses, opt.random_glimpse
  
  if opt.plan_route then
    local l_m = net.l_m
    local unitPixel = opt.unitPixels
    l_m:resize(N, E, T, 2):zero()
    l_m[{{}, {}, 1, 1}]:fill(-4/unitPixel)
    l_m[{{}, {}, 1, 2}]:fill(-4/unitPixel)
    l_m[{{}, {}, 2, 1}]:fill(-4/unitPixel)
    l_m[{{}, {}, 2, 2}]:fill(4/unitPixel)
    l_m[{{}, {}, 3, 1}]:fill(4/unitPixel)
    l_m[{{}, {}, 3, 2}]:fill(4/unitPixel)
    l_m[{{}, {}, 4, 1}]:fill(4/unitPixel)
    l_m[{{}, {}, 4, 2}]:fill(-4/unitPixel)
  end

  local score = net:forward(data)
  local score_view = score:view(N*E, -1)
  local label_view = label.new():resize(N*E):copy(label:view(N, 1):expand(N, E))
  local loss = criterior:forward(score_view, label_view)
  train_loss = loss
  table.insert(train_loss_history, train_loss)
  
  local _, predict = score_view:max(2)
  predict = predict:squeeze():type(type)
  
  local reward = torch.eq(predict, label_view):view(N, E):type(type)
  train_accuracy = reward:sum()/(N * E)
  table.insert(train_accuracy_history, train_accuracy)
  
--  print(net.l)
--  print(net.reward)
  table.insert(location_history, net.l_m:clone())
  table.insert(location_mean_history, net.l:clone())
  table.insert(reward_mean_history, rewardRollingMean)
  reward = (reward - rewardRollingMean) * opt.lamda
  rewardRollingMean = (rewardRollingMean * n + reward:sum()) / (n + reward:nElement())
  table.insert(reward_history, reward)
  
  n = n + reward:nElement()
--  local rewardBaseline = reward:sum(2) / E  -- N x 1
--  reward:add(-1, rewardBaseline:expand(N, E)):mul(opt.lamda) -- N x E
--  reward:mul(opt.lamda)
  pos_reward = torch.gt(reward, 0):sum()/(N * E)
  neg_reward = torch.lt(reward, 0):sum()/(N * E)
  zero_reward = torch.eq(reward, 0):sum()/(N * E)
  table.insert(pos_reward_history, pos_reward)
  table.insert(neg_reward_history, neg_reward)
  table.insert(zero_reward_history, zero_reward)
  
  local gradScore = criterior:backward(score_view, label_view):view(N, E, -1)
  net:backward(gradScore, reward)
  
  if opt.grad_clip > 0 then
    grads:clamp(-opt.grad_clip, opt.grad_clip)
  end
  
  return loss, grads
end

local validateIter = loader:iterator("validate")
local function calAccuracy(data, label)
  net:predict()
  local score = net:forward(data)[{{}, 1}] -- N x C
  local _, predict = score:max(2)
  predict = predict:squeeze():type(label:type())
  local hits = torch.eq(label, predict):sum()
  local nTotal = label:size(1)
    
  local loss = criterior:forward(score, label)

  return loss, hits/nTotal
end

local function calTrainEffect()
  net.planRoute = true
  net.l_m = location_history[#location_history]
  local lost, accuracy = calAccuracy(last_train_data, last_train_label)
  net.planRoute = net.planRoute
  return lost, accuracy
end

local function calValidateAccuracy()
  local data, label = validateIter.next_batch()
  data = data:type(type)
  label = label:type(type)
  return calAccuracy(data, label)
end

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
lfs.chdir(opt.checkpoint_dir)
if not path.exists(opt.source) then lfs.mkdir(opt.source) end
lfs.chdir(opt.source)
if not path.exists(opt.dataset) then lfs.mkdir(opt.dataset) end
lfs.chdir(opt.dataset)

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
  
  local _, loss = optim.adam(feval, params, optim_opt)
--  local _, loss = optim.sgd(feval, params, optim_opt)
--  local train_loss_after, train_accuracy_after = calTrainEffect()
  
--  print(loss[1], train_loss_after)
  local check_every = opt.checkpoint_every
  if i % iter_per_epoch == 0 or i == num_iterations then
--    print(net.l)
    
    checkpoint.iter = i
    checkpoint.epoch = epoch
    checkpoint.model = net
    checkpoint.opt = opt
    checkpoint.loss = loss[1]
    checkpoint.train_loss_after = train_loss_after
    local val_loss, val_accuracy = calValidateAccuracy("validate")
    checkpoint.validate_loss = val_loss
    checkpoint.validate_accuracy = val_accuracy
    
    checkpoint.train_loss_history = train_loss_history
    checkpoint.train_accuracy_history = train_accuracy_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_accuracy_history = val_accuracy_history
    checkpoint.reward_mean_history = reward_mean_history
    checkpoint.reward_history = reward_history
    checkpoint.location_history = location_history
    checkpoint.location_mean_history = location_mean_history
    print("i = ", i, " epoch = ", epoch, "train_loss = ", train_loss, "train_accuracy = ", train_accuracy, "val_loss = ", val_loss, "val_accuracy = ", val_accuracy)
--    print("i = ", i, " epoch = ", epoch, "train_loss = ", train_loss, "train_loss_after = ", train_loss_after, "train_accuracy = ", train_accuracy, "val_loss = ", val_loss, "val_accuracy = ", val_accuracy)
    table.insert(val_loss_history, val_loss)
    table.insert(val_accuracy_history, val_accuracy)
    table.insert(iterations, i)
    table.insert(epochs, epoch)
    local save_every = opt.save_every
    if (i % iter_per_epoch == 0) or i == num_iterations then
      local savefile = string.format('epoch%d.t7', checkpoint.epoch + start_epoch)
      torch.save(savefile, checkpoint)
    end
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