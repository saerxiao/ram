require 'torch'
require 'Ram'
require 'nn'
require 'optim'
require 'pl'

-- best so far for model with tanh, data size: 10000, batch = 100, r0=5e-3, decay_every = 3 epoch, decay_rate = 0.9, exploreation rate = 0.2
-- nepochs = 30, final val_loss = 1.42, val_accuracy = 65%
-- model without tanh has lower val_loss, seems similar val_accuracy

cmd = torch.CmdLine()
cmd:option('-source', 'nmist', 'directory for source data') --tinyshakespeare, qts, qsc
cmd:option('-validate_split', 0.9, 'sequence length')
cmd:option('-train_max_load', 1000, 'loading size')

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-glimpses', 4, 'number of glimpses')
cmd:option('-patch_size', 8)
cmd:option('-glimpse_hidden_size', 128)
cmd:option('-glimpse_output_size', 256)
cmd:option('-rnn_hidden_size', 256)
cmd:option('-nClasses', 10)
cmd:option('-location_gaussian_std', 0.03) -- 0.1
cmd:option('-exploration_rate', 0.15)
cmd:option('-episodes', 5, 'number of episodes')  -- 10 not better
cmd:option('-lamda', 0)
cmd:option('-random_glimpse', false, 'whether the glimpses are random')

-- training options
cmd:option('-batch_size', 20, 'batch size')
cmd:option('-nepochs', 50, 'number of epochs')
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-momentum', 0.9,'momentum')
cmd:option('-lr_decay_every', 5)   -- lr_decay_every=3, lr_decay_factor=0.9 seems to work well for train_max_load=10000 and batch_size=100
cmd:option('-lr_decay_factor', 0.7)
cmd:option('-grad_clip', 5)

cmd:option('-checkpoint_every', 10)
cmd:option('-save_every', 1000)
cmd:option('checkpoint_dir', 'checkpoint', 'dir for saving checkpoints')
cmd:option('checkpoint_subdir', 'ram_nmist', 'dir for saving checkpoints')

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

local net = Ram(opt)
local criterior = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterior = criterior:cuda()
  type = net:type()
end

local params, grads = net:getParameters()

local trainIter = loader:iterator("train")

local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label = trainIter.next_batch()
  data = data:type(type)
  label = label:type(type)
  local N, T, randomGlimpse = data:size(1), opt.glimpses, opt.random_glimpse
  
  local loss = 0
  if not randomGlimpse then
    local nEpisodes = opt.episodes
    local reward = label.new():zeros(N, nEpisodes)
    local l_m_all = label.new():resize(nEpisodes, N, T, 2)
    for ep = 1, nEpisodes do
      local score, l_m = net:forward(data)
--      loss = loss + criterior:forward(score, label)
      l_m_all[ep]:copy(l_m)
      local _, predict = score:max(2)
      predict = predict:squeeze():type(type)
      reward[{{}, ep}] = torch.eq(predict, label):type(type)
    end
  
    local rewardBaseline = reward:sum(2) / nEpisodes -- N x 1
    reward:add(-1, rewardBaseline:expand(N, nEpisodes)):mul(opt.lamda) -- N x nEpisodes
    
--    local gradsClassificationSum = grads.new():zeros(grads:size())
--    local gradsLocationSum = grads.new():zeros(grads:size())
  
    for ep = 1, nEpisodes do
      local score, _ = net:forward(data, l_m_all[ep])
      loss = loss + criterior:forward(score, label)
    
      local gradScore = criterior:backward(score, label)
      net:backward(gradScore, l_m_all[ep], reward[{{}, ep}])
--      net:backwardFromClassification(score, label, l_m_all[ep])
--      gradsClassificationSum:add(grads)
--    
--      grads:zero()
--      net:backwardFromLocation(reward[{{}, ep}], l_m_all[ep])
--      gradsLocationSum:add(grads)
    end
  
--    grads:zero():add(gradsClassificationSum):add(gradsLocationSum):div(nEpisodes)
    grads = grads/nEpisodes
    loss = loss/nEpisodes
  else
    local score, l_m = net:forward(data)
    loss = loss + criterior:forward(score, label)
    
    local gradScore = criterior:backward(score, label)
    net:backward(gradScore, l_m)
  end

  if opt.grad_clip > 0 then
    grads:clamp(-opt.grad_clip, opt.grad_clip)
  end
  
  return loss, grads
end

local function calAccuracy(split)
  local it = loader:iterator(split)
  local iter_per_epoch, T = loader.split_size[split], opt.glimpses
  local nTotal, hits, loss = 0, 0, 0
  for i = 1, iter_per_epoch do
    local data, label = it.next_batch()
    data = data:type(type)
    label = label:type(type)
    local score = net:forward(data) -- N x C
    local _, predict = score:max(2)
    predict = predict:squeeze():type(label:type())
    hits = hits + torch.eq(label, predict):sum()
    nTotal = nTotal + label:size(1)
    
    loss = loss + criterior:forward(score, label)
  end
  return loss/iter_per_epoch, hits/nTotal
end

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
lfs.chdir(opt.checkpoint_dir)
if not path.exists(opt.checkpoint_subdir) then lfs.mkdir(opt.checkpoint_subdir) end
lfs.chdir(opt.checkpoint_subdir)

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
  
  local check_every = opt.checkpoint_every
  if i % iter_per_epoch == 0 or (check_every > 0 and i % check_every == 0) or i == num_iterations then
    checkpoint.iter = i
    checkpoint.epoch = epoch
    checkpoint.model = net
    checkpoint.opt = opt
    checkpoint.loss = loss[1]
    local val_loss, val_accuracy = calAccuracy("validate")
    checkpoint.validate_loss = val_loss
    checkpoint.validate_accuracy = val_accuracy
    print("i = ", i, " epoch = ", epoch, "val_loss = ", val_loss, "val_accuracy = ", val_accuracy)
    table.insert(val_loss_history, val_loss)
    table.insert(val_accuracy_history, val_accuracy)
    table.insert(iterations, i)
    table.insert(epochs, epoch)
    local save_every = opt.save_every
    if (i % iter_per_epoch == 0) or i == num_iterations then
      local savefile = string.format('epoch%d_i%d.t7', checkpoint.epoch, i)
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