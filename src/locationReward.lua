require 'torch'
require 'Ram1_vr'
require 'dp'
require 'rnn'
require 'optim'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', 'checkpoint/mnist.t7/32x32/epoch800.t7', 'path to a previously saved model')
--cmd:option('--xpPath', 'saved-model/epoch1.t7', 'path to a previously saved model')
cmd:option('--cuda', true, 'model was saved with cuda')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)
--model = xp.model
--ra = model:findModules('Ram1')[1]
--print(ra.reward)

reward_history = xp.reward_history
location_history = xp.location_history
local imageW = 32
local N, T = location_history[1]:size(1), location_history[1]:size(2)
--print(#reward_history)
location_mean_history = xp.location_mean_history

local nIters = #reward_history
print(nIters)
local n = nIters - 500
local endN = nIters
local step = 100

repeat
  heatmaps = {}
  local from = n
  local to = (from + step - 1 < nIters) and from+step-1 or nIters
  
  for iter = from, to do
    local location = location_history[iter]:squeeze()
    local reward = reward_history[iter]
--    print(location_mean_history[iter])
--    print(location)
--    local heatmap = reward.new():resize(imageW, imageW):zero()
    local  iterIndex = iter-from+1
    for i = 1, N do
      for t = 1, T do
        local heatmaps_t = heatmaps[t]
        if not heatmaps_t then
          heatmaps_t = {}
          heatmaps[t] = heatmaps_t
        end
        
        local heatmap = heatmaps_t[iterIndex]
        if not heatmap then
          heatmap = reward.new():resize(imageW, imageW):zero()
--          table.insert(heatmaps_t, heatmap)
          heatmaps_t[iterIndex] = heatmap
        end
        x, y = location[i][t][1], location[i][t][2]
        x, y = (x+1)/2, (y+1)/2
        x, y = x*(imageW-1)+1, y*(imageW-1)+1
        heatmap[x][y] = heatmap[x][y] + reward[i]
      end
    end
--print(torch.eq(heatmap,0):sum())
--print(torch.gt(heatmap,0):sum())
--print(torch.lt(heatmap,0):sum()) 
  for t = 1, T do
--    local iterCnt = #heatmaps[t]
    heatmaps[t][iterIndex]:maskedFill(torch.gt(heatmaps[t][iterIndex], 0), 1)
    heatmaps[t][iterIndex]:maskedFill(torch.lt(heatmaps[t][iterIndex], 0), -1)
  end

--print(heatmap)
--    gnuplot.imagesc(heatmap)
--    table.insert(heatmaps, heatmap)
  end
  n = to + 1
  for t = 1, T do
    local g = image.toDisplayTensor{input=heatmaps[t],nrow=10,padding=3}
    image.save("glimpse/rl_" .. from .. "_" .. to .. "_" .. t ..".png", g)
  end
until n > endN

--repeat
--  heatmaps = {}
--  local from = n
--  local to = (from + step - 1 < nIters) and from+step-1 or nIters
--  
--  for iter = from, to do
--    local location = location_history[iter]:squeeze()
--    local reward = reward_history[iter]
----    print(location_mean_history[iter])
----    print(reward)
--    local heatmap = reward.new():resize(imageW, imageW):zero()
--    for i = 1, N do
--      for t = 1, T do
--        x, y = location[i][t][1], location[i][t][2]
--        x, y = (x+1)/2, (y+1)/2
--        x, y = x*(imageW-1)+1, y*(imageW-1)+1
--        heatmap[x][y] = heatmap[x][y] + reward[i]
--      end
--    end
----print(torch.eq(heatmap,0):sum())
----print(torch.gt(heatmap,0):sum())
----print(torch.lt(heatmap,0):sum())
--heatmap:maskedFill(torch.gt(heatmap, 0), 1)
--heatmap:maskedFill(torch.lt(heatmap, 0), -1)
----print(heatmap)
----  gnuplot.imagesc(heatmap)
--    table.insert(heatmaps, heatmap)
--  end
--  n = to + 1
--  local g = image.toDisplayTensor{input=heatmaps,nrow=10,padding=3}
--  image.save("glimpse/reward_location_" .. from .. "_" .. to .. ".png", g)
--until n > endN

