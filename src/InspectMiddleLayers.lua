require 'torch'
require 'Ram1_vr'
require 'dp'
require 'rnn'
require 'optim'
require 'lfs'
require 'gnuplot'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--dir', 'checkpoint/mnist.t7/32x32/', 'dir of the files')
cmd:option('--xpPath', 'checkpoint/mnist.t7/32x32/epoch100.t7', 'path to a previously saved model')
--cmd:option('--xpPath', 'saved-model/epoch1.t7', 'path to a previously saved model')
cmd:option('--cuda', true, 'model was saved with cuda')
cmd:option('-glimpses', 4, 'number of glimpses')
cmd:option('-glimpseOutputSize', 256)
cmd:option('--batchSize', 20, 'batch size')
cmd:text()
local opt = cmd:parse(arg or {})

local N, D = opt.batchSize, opt.glimpseOutputSize
local T, imageW = 4, 32
-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

local function find_epoch(str)
--    return tonumber(string.sub(str, string.len('epoch') + 1, string.find(str,'%.') - 1))
  return tonumber(string.sub(str, string.find(str, 'epoch') + 5, string.len(str) - 3))
end
  
local function getFiles(dir)
--  lfs.chdir(dir)
  print(lfs.currentdir())
--  local dir = 'checkpoint/'
  local files = {}
  for file in lfs.dir(dir) do
    if lfs.attributes(dir .. "/" .. file, "mode") == "file" and string.find(file, 'epoch') then
      table.insert(files,dir .. "/" .. file)
    end
  end
--  for file in lfs.dir('.') do
--    if lfs.attributes(file, "mode") == "file" and string.find(file, 'epoch') then
--      table.insert(files, file)
--    end
--  end

  -- sort files by iteration number
  

  table.sort(files, function (a,b) return find_epoch(a) < find_epoch(b) end)
  
  return files
end

local function rewardLoc(xpPath, epoch)
  local xp = torch.load(xpPath)
  local model = xp.model
--model = xp:model().module
  local ra = model:findModules('Ram1_vr')[1]
--print(ra.reward)

  local glimpses = ra.glimpses
  local rnns = ra.rnns
  local locationNets = ra.locationNets
  local l_m = ra.l_m
  local reward = ra.reward
  local heatmaps = {}
  local ll = reward.new():resize(N, T, 2)
  local bn = reward.new():resize(N,T,2)
  for t = 1, T do
--  local glimpse = glimpses[t]
--  print(glimpse.output)
--  print(ra.g[{{}, t}])
    local rnn = rnns[t]
--  print(rnn.output)
    local locationNet = locationNets[t]
  ll[{{}, t}] = locationNet.modules[1].output
--  bn[{{}, t}] = locationNet.modules[2].output
--  print(locationNet.modules[1].output)
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

--  print(ll)
  --print(bn)
  local g = image.toDisplayTensor{input=heatmaps,nrow=T,padding=3}
  g = image.scale(g, g:size(3)*5, g:size(2)*5)
  image.save("glimpse/a_rl_" .. epoch ..".png", g)
end


local files = getFiles(opt.dir)
for i = 1, #files do
  if i < 101 then
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
