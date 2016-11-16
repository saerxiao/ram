require 'torch'
require 'Ram1'
require 'nn'
require 'optim'
require 'pl'

cmd = torch.CmdLine()
cmd:option('-source', 'nmist', 'directory for source data') --tinyshakespeare, qts, qsc
cmd:option('-validate_split', 0.9, 'sequence length')
cmd:option('-train_max_load', 10000, 'loading size')

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-glimpses', 4, 'number of glimpses')
cmd:option('-patch_size', 8)
cmd:option('-glimpse_hidden_size', 128)
cmd:option('-glimpse_output_size', 256)
cmd:option('-rnn_hidden_size', 256)
cmd:option('-nClasses', 10)
cmd:option('-location_gaussian_std', 0.1) -- 0.1
cmd:option('-exploration_rate', 0.15)
cmd:option('-episodes', 5, 'number of episodes')  -- 10 not better
cmd:option('-lamda', 10)
cmd:option('-random_glimpse', false, 'whether the glimpses are random')

-- training options
cmd:option('-batch_size', 20, 'batch size')
cmd:option('-nepochs', 10, 'number of epochs')
cmd:option('-learning_rate',5e-4,'learning rate')
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

local net = Ram1(opt)
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
  net:isTrain()
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label = trainIter.next_batch()
  data = data:type(type)
  label = label:type(type)
  local N, E, T, randomGlimpse = data:size(1), opt.episodes, opt.glimpses, opt.random_glimpse
  
  local score = net:forward(data)
  local score_view = score:view(N*E, -1)
  local label_view = label.new():resize(N*E):copy(label:view(N, 1):expand(N, E))
  local loss = criterior:forward(score_view, label_view)
  
  local _, predict = score_view:max(2)
  predict = predict:squeeze():type(type)
  
  local reward = torch.eq(predict, label_view):view(N, E):type(type)
  
  local rewardBaseline = reward:sum(2) / E  -- N x 1
  reward:add(-1, rewardBaseline:expand(N, E)):mul(opt.lamda) -- N x E
  
  local gradScore = criterior:backward(score_view, label_view):view(N, E, -1)
  net:backward(gradScore, reward)
  
  return loss, grads
end

local function calAccuracy(split)
  net:predict()
  local it = loader:iterator(split)
  local iter_per_epoch, T = loader.split_size[split], opt.glimpses
  local nTotal, hits, loss = 0, 0, 0
  for i = 1, iter_per_epoch do
    local data, label = it.next_batch()
    data = data:type(type)
    label = label:type(type)
    local score = net:forward(data)[{{}, 1}] -- N x C
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