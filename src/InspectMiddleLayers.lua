package.path = package.path .. ";/home/saxiao/ram/src/?.lua"

require 'torch'
require 'Ram1_vr'
require 'dp'
require 'rnn'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'RA1'
require 'Rnn'
require 'VanillaRnn'
require 'Recurrent'
require 'Recursor'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--myScript', false, 'use my implementation')
cmd:option('--myModel', false, 'use my implementation')
cmd:option('--raModule', 'nn.RA1', 'name of the reccurent attention module')  --nn. RA1, nn.RecurrentAttention
cmd:option('--dir', 'saved-model', 'dir of the files') -- saved-model checkpoint/mnist.t7/32x32/
cmd:option('-glimpses', 4, 'number of glimpses')
cmd:option('-glimpseOutputSize', 256)
cmd:option('--batchSize', 20, 'batch size')
cmd:option('--imageWidth', 32, 'batch size')
cmd:option('--cuda', true, 'model was saved with cuda')
cmd:text()
local opt = cmd:parse(arg or {})

local N, D = opt.batchSize, opt.glimpseOutputSize
local T, imageW, raModule = opt.glimpses, opt.imageWidth, opt.raModule

if opt.cuda then
   require 'cunn'
end

local function find_epoch(str)
  return tonumber(string.sub(str, string.find(str, 'epoch') + 5, string.len(str) - 3))
end
  
local function getFiles(dir)
  print(lfs.currentdir())
  local files = {}
  for file in lfs.dir(dir) do
    if lfs.attributes(dir .. "/" .. file, "mode") == "file" and string.find(file, 'epoch') then
      table.insert(files,dir .. "/" .. file)
    end
  end

  table.sort(files, function (a,b) return find_epoch(a) < find_epoch(b) end)
  
  return files
end

local function erModelRewardLoc(model)
  local ra = model:findModules(raModule)[1]
  local l_m = torch.Tensor(opt.batchSize,opt.glimpses,2):type(ra:type())
  for j,location in ipairs(ra.actions) do
    l_m[{{},j}] = location
  end
  local rn = ra.action:getStepModule(1):findModules('nn.ReinforceNormal')[1]
  local reward = rn.reward
  return l_m, reward
end

local function myModelRewardLoc(model)
  local ra = model:findModules('Ram1_vr')[1]
  return ra.l_m, ra.reward
end

local function rewardLoc(xpPath, epoch)
  local xp = torch.load(xpPath)
  local model = nil
  if opt.myScript then
    model = xp.model
  else
    model = xp:model().module
  end
  
  local l_m, reward = nil, nil
  if opt.myModel then
    l_m, reward = myModelRewardLoc(model)
  else
    l_m, reward = erModelRewardLoc(model)
  end
  
  print(l_m)
  local heatmaps = {}
  local ll = reward.new():resize(N, T, 2)
  local bn = reward.new():resize(N,T,2)
  for t = 1, T do
    local N = reward:size(1)
    local heatmap = reward.new():resize(imageW, imageW):zero()
    for i = 1, N do
      local x, y = l_m[i][t][1], l_m[i][t][2]
      x, y = (x+1)/2, (y+1)/2
      x, y = x*(imageW-1)+1, y*(imageW-1)+1
      heatmap[x][y] = heatmap[x][y] + reward[i]
    end
    heatmap:maskedFill(torch.gt(heatmap, 0), 1)
    heatmap:maskedFill(torch.lt(heatmap, 0), -1)
    heatmaps[t] = heatmap
  end

  local g = image.toDisplayTensor{input=heatmaps,nrow=T,padding=3}
  g = image.scale(g, g:size(3)*5, g:size(2)*5)
  image.save("glimpse/a_rl_" .. epoch ..".png", g)
end


local files = getFiles(opt.dir)
for i = 1, #files do
  if i >83 and i < 220 then
    rewardLoc(files[i], i)
    print('printed epoch ' .. i)
  end
end

--
--local files = getFiles()
--local xg = torch.Tensor():resize(T, N*D*(#files))
--local yg = torch.Tensor():resize(T, N*D*(#files))
--local xl = torch.Tensor():resize(T, N*2*(#files))
--local yl = torch.Tensor():resize(T, N*2*(#files))
--for i = 1, #files do
--  local epoch = find_epoch(files[i])
--  local f = torch.load(files[i])
--  local from, to = (i-1)*N*D + 1, i*N*D
--  local ra = f.model:findModules('Ram1_vr')[1]
--  xg[{{}, {(i-1)*N*D, i*N*D}}]:fill(epoch)
--  yg[{{}, {(i-1)*N*D, i*N*D}}]:copy(ra.g:transpose(1,2))
--  xl[{t, {(i-1)*N*2, i*N*2}}]:fill(epoch)
--  for t = 1, T do
--    yl[{t, {(i-1)*N*2, i*N*2}}]:copy(ra.locationNets[t].modules[1].output)
--  end
--end
----gnuplot.plot(xl, yl, '+')
--gnuplot.plot(xg, yg, '+')
