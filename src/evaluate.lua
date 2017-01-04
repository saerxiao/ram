require 'dp'
require 'rnn'
require 'optim'
require 'Ram2'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', 'checkpoint/mnist.t7/32x32/epoch1.t7', 'path to a previously saved model')
cmd:option('--cuda', true, 'model was saved with cuda')
cmd:option('--evalTest', false, 'model was saved with cuda')
cmd:option('--stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('--overwrite', false, 'overwrite checkpoint')
cmd:option('--imageDir', 'iter1/', 'overwrite checkpoint')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)
model = xp:model().module 
tester = xp:tester() or xp:validator() -- dp.Evaluator
tester:sampler()._epoch_size = nil
conf = tester:feedback() -- dp.Confusion
cm = conf._cm -- optim.ConfusionMatrix

print("Last evaluation of "..(xp:tester() and 'test' or 'valid').." set :")
print(cm)

if opt.dataset == 'TranslatedMnist' then
   ds = torch.checkpoint(
      paths.concat(dp.DATA_DIR, 'checkpoint/dp.TranslatedMnist_test.t7'),
      function() 
         local ds = dp[opt.dataset]{load_all=false} 
         ds:loadTest()
         return ds
         end, 
      opt.overwrite
   )
else
   ds = dp[opt.dataset]()
end

ra = model:findModules('Ram2')[1]
print(ra.reward)
--sg = model:findModules('nn.SpatialGlimpse')[1]

-- stochastic or deterministic
--for i=1,#ra.actions do
--   local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
--   rn.stochastic = opt.stochastic
--end

local rnns = ra.rnns

if opt.evalTest then
   conf:reset()
   tester:propagateEpoch(ds:testSet())

   print((opt.stochastic and "Stochastic" or "Deterministic") .. "evaluation of test set :")
   print(cm)
end

inputs = ds:get('test','inputs')
targets = ds:get('test','targets', 'b')

input = inputs:narrow(1,1,10)
model:training() -- otherwise the rnn doesn't save intermediate time-step states
--if not opt.stochastic then
--   for i=1,#ra.actions do
--      local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
--      rn.stdev = 0 -- deterministic
--   end
--end
output = model:forward(input)

function drawBox(img, bbox, channel)
    channel = channel or 1

    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    x1, y1 = math.max(1, x1), math.max(1, y1)
    x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

    local max = img:max()

    for i=x1,x2 do
        img[channel][y1][i] = max
        img[channel][y2][i] = max
    end
    for i=y1,y2 do
        img[channel][i][x1] = max
        img[channel][i][x2] = max
    end

    return img
end

--locations = ra.actions
local T = ra.T
local l_m, patch_tensor = ra.l_m, ra.patch -- N x T x 2
--print(l_m)
l_m = l_m:transpose(1, 2)
patch_tensor = patch_tensor:transpose(1, 2)

input = nn.Convert(ds:ioShapes(),'bchw'):forward(input)
glimpses = {}
patches = {}

params = nil
for t = 1, T do
  local glimpse = glimpses[t] or {}
  glimpses[t] = glimpse
  local patch = patches[t] or {}
  patches[t] = patch
  local location = l_m[t]
  for i = 1, input:size(1) do
    local img = input[i]
    local xy = location[i]
    -- (-1,-1) top left corner, (1,1) bottom right corner of image
    local x, y = xy:select(1,1), xy:select(1,2)
    -- (0,0), (1,1)
    x, y = (x+1)/2, (y+1)/2
    -- (1,1), (input:size(3), input:size(4))
    x, y = x*(input:size(3)-1)+1, y*(input:size(4)-1)+1
    
    local gimg = img:clone()
    local size = ra.patchSize
    local bbox = {y-size/2, x-size/2, size, size}
    drawBox(gimg, bbox, 1)
    glimpse[i] = gimg
    
--    patch[i] = patch_tensor[t][i]:view(1, size, size)
    patch[i] = image.scale(img:clone():float(), patch_tensor[t][i]:view(1, size, size):float())
    
    collectgarbage()
  end
end

paths.mkdir('glimpse')
--lfs.chdir('glimpse')
--paths.mkdir('100')
--lfs.chdir('100')
--paths.mkdir(opt.imageDir)
for j,glimpse in ipairs(glimpses) do
   local g = image.toDisplayTensor{input=glimpse,nrow=10,padding=3}
   local p = image.toDisplayTensor{input=patches[j],nrow=10,padding=3}
--   image.save(opt.imageDir .. "glimpse"..j..".png", g)
--   image.save(opt.imageDir ..  "patch"..j..".png", p)
   image.save("glimpse/glimpse"..j..".png", g)
   image.save("glimpse/patch"..j..".png", p)
end