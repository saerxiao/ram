require 'torch'
require 'gnuplot'

local transform = {}

local trainFileName = 'mnist.t7/train_32x32.t7'
local testFileName = 'mnist.t7/test_32x32.t7'

local function generateOffCenter(iFile, oFile, L)
  local f = torch.load(iFile, 'ascii')
  local data = f.data:type(torch.getdefaulttensortype())
  local labels = f.labels
  
  data = data[{{1,10}}]
  labels = labels[{{1, 10}}]
  local N, C, l = data:size(1), data:size(2), data:size(3)
  local generated = torch.Tensor(N, C, L, L) 
  for i = 1, N do
    local x = torch.random(l/2, L-l/2)
    local y = torch.random(l/2, L-l/2)
    local xlow = x - l/2 + 1
    local xhi = x + l/2
    local ylow = y - l/2 + 1
    local yhi = y + l/2
    generated[i][1]:sub(xlow, xhi, ylow, yhi):copy(data[i][1])
--    gnuplot.raw('set multiplot layout 1,2')
--    gnuplot.imagesc(data[i][1])
--    gnuplot.imagesc(generated[i][1])
  end
  local output = {}
  output.data = generated
  output.labels = labels
  torch.save(oFile, output)
end

function transform.offCenter(L)
  local trainOFile = 'mnist.t7/train_offcenter_' .. L .. 'x' .. L .. '_10.t7'
  generateOffCenter(trainFileName, trainOFile, L)
  
  local testOFile = 'mnist.t7/test_offcenter_' .. L .. 'x' .. L .. '_10.t7'
  generateOffCenter(testFileName, testOFile, L)
end

return transform