require 'torch'
require 'nngraph'

--local glimpse = torch.class('Glimpse')
local glimpse = {}

function glimpse.createNet(patchSize, glimpseHiddenSize, glimpseOutputSize, r)
  local patchSize, H, O = patchSize, glimpseHiddenSize, glimpseOutputSize
--  self.H = glimpseHiddenSize
--  self.O = glimpseOutputSize
--  self.exploration_rate = r
--  
--  local H, O = self.H, self.O
  local graph = nn.Linear(patchSize*patchSize, H)()
  local location = nn.Linear(2, H)()
  local gh = nn.Linear(H, O)(nn.ReLU()(graph))
  local fh = nn.Linear(H, O )(nn.ReLU()(location))
  local madd = nn.CAddTable()({gh, fh})
  local g = nn.ReLU()(madd)
  return nn.gModule({graph, location}, {g})
--  self.patch = torch.Tensor()
--  return gnet
end

function glimpse.computePatch(image, l_m, patchSize, exploration_rate)
  local N, nChannel, imageSizeX, imageSizeY = image:size(1), image:size(2), image:size(3), image:size(4)
  assert(nChannel == 1, 'image must only have 1 color channel')
  assert(imageSizeX == imageSizeY, 'image must be square')
  local pixel_per_unit_length = exploration_rate * imageSizeX
  local low = (torch.floor(l_m * pixel_per_unit_length) - patchSize / 2 + 1) + imageSizeX / 2 
--  low[{{}, 1}] = (low[{{}, 1}] + imageSizeX / 2):clamp(1, imageSizeX)
--  low[{{}, 2}] = (low[{{}, 2}] + imageSizeY / 2):clamp(1, imageSizeY)
  local hi = (torch.floor(l_m * pixel_per_unit_length) + patchSize / 2) + imageSizeX / 2
--  hi[{{}, 1}] = (hi[{{}, 1}] + imageSizeX / 2):clamp(1, imageSizeX)
--  hi[{{}, 2}] = (hi[{{}, 2}] + imageSizeY / 2):clamp(1, imageSizeY)
--  local mask = low:le(1)
--  low:maskedFill(mask, 1)
--  hi:maskedFill(mask, self.patchSize)
  
  local patch = image.new():resize(N, patchSize, patchSize)
  for i = 1, image:size(1) do
    if low[i][1] > 0 and low[i][2] > 0  and hi[i][1] <= imageSizeX and hi[i][2] <= imageSizeY then
      patch[{i}] = image[{i, 1}]:sub(low[i][1], hi[i][1], low[i][2], hi[i][2])
    else
      patch[{i}]:fill(0)
    end
  end
  return patch:view(N, patchSize * patchSize)
end

--function glimpse:forward(image, l_m)
--  local patch = self:computePatch(image, l_m):view(self.patchSize*self.patchSize)
--  return self.gnet:forward({patch, l_m})
--end
--
--function glimpse:backward(l_m, gradOutput)
--  local patch = self.patch:view(self.patchSize*self.patchSize)
--  return self.gnet:backward({patch, l_m}, gradOutput)
--end

return glimpse