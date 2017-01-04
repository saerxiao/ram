require 'torch'
require 'nn'
require 'optim'
require 'Ram1'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', 'checkpoint/mnist.t7/32x32/epoch200.t7', 'path to a previously saved model')
cmd:option('--cuda', true, 'model was saved with cuda')
cmd:option('--imageW', 32, 'input image width')
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

checkpoint = torch.load(opt.xpPath)
model  = checkpoint.model

print(model.l[{{}, 1}])

local rnns = model.rnns

if opt.evalTest then
   conf:reset()
   tester:propagateEpoch(ds:testSet())

   print((opt.stochastic and "Stochastic" or "Deterministic") .. "evaluation of test set :")
   print(cm)
end

function drawPatch(img, bbox, patch)
    local imageW, patchSize = opt.imageW, model.patchSize
    local x1, y1 = math.floor(bbox[1]), math.floor(bbox[2])
    local x2, y2 = x1 + bbox[3] - 1, y1 + bbox[4] - 1
--    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
--    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    local px1, py1, px2, py2 = 1, 1, patchSize, patchSize
    if x1 < 1 then
      px1 = 1 + (1 - x1 + 1) - 1
    end
    if y1 < 1 then
      py1 = 1 + (1 - y1 + 1) - 1
    end
    x1, y1 = math.max(1, x1), math.max(1, y1)
    if x2 > imageW then
      px2 = 1 + (imageW - x1 + 1) - 1
    end
    if y2 > imageW then
      py2 = 1 + (imageW - y1 + 1) - 1
    end
    x2, y2 = math.min(imageW, x2), math.min(imageW, y2)
    
    img:sub(y1,y2, x1,x2):copy(patch:sub(px1,px2,py1,py2))

    local max = 256

    for i=x1,x2 do
        img[y1][i] = max
        img[y2][i] = max
    end
    for i=y1,y2 do
        img[i][x1] = max
        img[i][x2] = max
    end

    return img
end

--locations = ra.actions
local T, imageW = model.T, opt.imageW
local l_m, patch_tensor = model.l_m[{{},1}], model.patch[{{}, 1}] -- N x T x 2
--print(l_m)
l_m = l_m:transpose(1, 2)
patch_tensor = patch_tensor:transpose(1, 2)

glimpses = {}
patches = {}

params = nil
for t = 1, T do
  local glimpse = glimpses[t] or {}
  glimpses[t] = glimpse
  local patch = patches[t] or {}
  patches[t] = patch
  local location = l_m[t]
  for i = 1, 10 do
    local xy = location[i]
    -- (-1,-1) top left corner, (1,1) bottom right corner of image
    local x, y = xy:select(1,1), xy:select(1,2)
    -- (0,0), (1,1)
    x, y = (x+1)/2, (y+1)/2
    -- (1,1), (input:size(3), input:size(4))
    x, y = x*(imageW-1)+1, y*(imageW-1)+1
    
    local gimg = xy.new():resize(imageW, imageW):zero()
    local size = model.patchSize
    local bbox = {y-size/2, x-size/2, size, size}
    drawPatch(gimg, bbox, patch_tensor[t][i]:view(1, size, size)[1])
    glimpse[i] = gimg
    
--    patch[i] = patch_tensor[t][i]:view(1, size, size)
--    patch[i] = image.scale(img:clone():float(), patch_tensor[t][i]:view(1, size, size):float())
    
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
--   local p = image.toDisplayTensor{input=patches[j],nrow=10,padding=3}
   image.save("glimpse/glimpse"..j..".png", g)
--   image.save("glimpse/patch"..j..".png", p)
end