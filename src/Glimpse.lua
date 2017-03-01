require 'torch'
require 'nngraph'
require 'image'

--local glimpse = torch.class('Glimpse')
local glimpse = {}

function glimpse.createNet(patchSize, glimpseHiddenSize, glimpseOutputSize, scales)
  local S = scales and scales or 1
  local patchSize, H, O = patchSize, glimpseHiddenSize, glimpseOutputSize
  local graph = nn.Linear(S*patchSize*patchSize, H)()
  local location = nn.Linear(2, H)()
  local gh = nn.Linear(H, O)(nn.ReLU()(graph))
  local fh = nn.Linear(H, O )(nn.ReLU()(location))
  local madd = nn.CAddTable()({gh, fh})
  local g = nn.ReLU()(madd)
  return nn.gModule({graph, location}, {g})
end

function glimpse.createNet1(patchSize, glimpseHiddenSize, locatorHiddenSize, glimpseOutputSize, scales)
  local S = scales and scales or 1
  local patchSize, H_g, H_l,O = patchSize, glimpseHiddenSize, locatorHiddenSize, glimpseOutputSize
  local graph = nn.Linear(S*patchSize*patchSize, H_g)()
  local location = nn.Linear(2, H_l)()
  local gh = nn.Linear(H_g, O)(nn.ReLU()(graph))
  local fh = nn.Linear(H_l, O)(nn.ReLU()(location))
  local madd = nn.CAddTable()({gh, fh})
  local g = nn.ReLU()(madd)
  return nn.gModule({graph, location}, {g})
end

function glimpse.computePatch(image, l_m, patchSize)
  local N, nChannel, imageSizeX, imageSizeY = image:size(1), image:size(2), image:size(3), image:size(4)
--  l_m:mul(unitPixels*2/imageSizeX)
  assert(nChannel == 1, 'image must only have 1 color channel')
  assert(imageSizeX == imageSizeY, 'image must be square')
--  local low = (torch.floor(l_m * unitPixels) - patchSize / 2 + 1) + imageSizeX / 2 
--  local hi = (torch.floor(l_m * unitPixels) + patchSize / 2) + imageSizeX / 2
  local center = torch.floor((l_m+1) / 2 * imageSizeX)
  local low = center - patchSize / 2 + 1
  local hi = center + patchSize / 2
  low:clamp(1, imageSizeX)
  hi:clamp(1, imageSizeX)
  
  local patch = image.new():resize(N, patchSize, patchSize):fill(0)
  for i = 1, N do
    local l = hi[i] - low[i] + 1
    local pxlow = (l[1] < patchSize and low[i][1] == 1) and patchSize - l[1] + 1 or 1
    local pxhi = (l[1] < patchSize and hi[i][1] == imageSizeX) and l[1]  or patchSize
    local pylow = (l[2] < patchSize and low[i][2] == 1) and patchSize - l[2] + 1 or 1
    local pyhi = (l[2] < patchSize and hi[i][2] == imageSizeX) and l[2]  or patchSize
    patch[{i}]:sub(pxlow, pxhi, pylow, pyhi):copy(image[{i, 1}]:sub(low[i][1], hi[i][1], low[i][2], hi[i][2]))
  end
  return patch:view(N, patchSize * patchSize)
end

function glimpse.computePatchRam3(image, l_m, patchSize, unitPixels)
  local N, nChannel, imageSizeX, imageSizeY = image:size(1), image:size(2), image:size(3), image:size(4)
  assert(nChannel == 1, 'image must only have 1 color channel')
  assert(imageSizeX == imageSizeY, 'image must be square')
--  local low = (torch.floor(l_m * unitPixels) - patchSize / 2 + 1) + imageSizeX / 2 
--  local hi = (torch.floor(l_m * unitPixels) + patchSize / 2) + imageSizeX / 2
  local center = torch.floor((l_m+1) / 2 * imageSizeX)
  local low = center - patchSize / 2 + 1
  local hi = center + patchSize / 2
  low:clamp(1, imageSizeX)
  hi:clamp(1, imageSizeX)
  
  local patch = image.new():resize(N, patchSize, patchSize):fill(0)
  for i = 1, N do
    local l = hi[i] - low[i] + 1
    local pxlow = (l[1] < patchSize and low[i][1] == 1) and patchSize - l[1] + 1 or 1
    local pxhi = (l[1] < patchSize and hi[i][1] == imageSizeX) and l[1]  or patchSize
    local pylow = (l[2] < patchSize and low[i][2] == 1) and patchSize - l[2] + 1 or 1
    local pyhi = (l[2] < patchSize and hi[i][2] == imageSizeX) and l[2]  or patchSize
    patch[{i}]:sub(pxlow, pxhi, pylow, pyhi):copy(image[{i, 1}]:sub(low[i][1], hi[i][1], low[i][2], hi[i][2]))
  end
  return patch:view(N, patchSize * patchSize)
end

function glimpse.computePatchMultipleScale1(src, l_m, P, r, nScales)
  local N, nChannel, imageSizeX, imageSizeY = src:size(1), src:size(2), src:size(3), src:size(4)
  assert(nChannel == 1, 'image must only have 1 color channel')
  assert(imageSizeX == imageSizeY, 'image must be square')
  local l_m_pixel = torch.floor(l_m * r * imageSizeX) + imageSizeX / 2 
  local patch = src.new():resize(N, nScales, P, P):fill(0)
  local buffer = src.new()
  local ps = P
  for s = 1, nScales do
    if ps > imageSizeX then ps = imageSizeX end
    buffer:resize(ps, ps)
    local low = l_m_pixel - ps / 2 + 1
    local hi = l_m_pixel + ps / 2
    low:clamp(1, imageSizeX)
    hi:clamp(1, imageSizeX)
    for i = 1, N do
      local l = hi[i] - low[i] + 1
      local pxlow = (l[1] < ps and low[i][1] == 1) and ps - l[1] + 1 or 1
      local pxhi = (l[1] < ps and hi[i][1] == imageSizeX) and l[1]  or ps
      local pylow = (l[2] < ps and low[i][2] == 1) and ps - l[2] + 1 or 1
      local pyhi = (l[2] < ps and hi[i][2] == imageSizeX) and l[2]  or ps
      buffer:fill(0)
      buffer:sub(pxlow, pxhi, pylow, pyhi):copy(src[{i,1}]:sub(low[i][1], hi[i][1], low[i][2], hi[i][2]))
      patch[{i, s}]:copy(image.scale(buffer, P))
    end
    ps = ps * 2
  end
  return patch:view(N, nScales, P, P)
end

function glimpse.computePatchMultipleScale(image, l_m, P, r, steps)
  local steps = steps and steps or torch.Tensor{4,8}
  local N, S = image:size(1), steps:size(1)
  local patch = image.new():resize(N, 1 + S, P, P)
  patch[{{}, 1}]:copy(glimpse.computePatch(image,l_m,P,r))
  for s = 1, S do
    patch[{{}, s+1}]:copy(glimpse.downSample(image,l_m,P,r,steps[s]))
  end    
  return patch:view(N, -1)
end

function glimpse.downSample(image, l_m, P, r, step)
  local N, nChannel, imageSizeX, imageSizeY = image:size(1), image:size(2), image:size(3), image:size(4)
  assert(nChannel == 1, 'image must only have 1 color channel')
  assert(imageSizeX == imageSizeY, 'image must be square')
  local pixel_per_unit_length = r * imageSizeX
  local patch = image.new():resize(N, P, P):fill(0)
  local low = image.new():resize(N,2):fill(0)
  local hi = image.new():resize(N,2):fill(0)
  for i = 1, P do
    low[{{}, 1}] = l_m[{{}, 1}] * pixel_per_unit_length + (i - P/2 - 1) * step + 1 + imageSizeX / 2
    hi[{{}, 1}] = l_m[{{}, 1}] * pixel_per_unit_length + (i - P/2) * step + imageSizeX / 2 
    for j = 1, P do
      low[{{}, 2}] = l_m[{{}, 2}] * pixel_per_unit_length + (j - P/2 - 1) * step + 1 + imageSizeX / 2 
      hi[{{}, 2}] = l_m[{{}, 2}] * pixel_per_unit_length + (j - P/2) * step + imageSizeX / 2
--      low:maskedFill(low:le(1), 1)
--      hi:maskedFill(hi:gt(imageSizeX), imageSizeX)
      low:clamp(1, imageSizeX)
      hi:clamp(1, imageSizeX)
      for n = 1, N do
        patch[n][i][j] = image[n][1]:sub(low[n][1], hi[n][1], low[n][2], hi[n][2]):sum() / (P * P)
      end
    end
  end
  return patch:view(N, P * P)
end

return glimpse