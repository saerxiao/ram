require 'torch'
require 'nngraph'
require 'gnuplot'
--require 'image'

local glimpseUtil = require 'Glimpse'
local glimpse = glimpseUtil.createNet1(8, 128, 128,
    256, 1)

--local x1 = torch.randn(40):mul(100)
--local y1 = torch.randn(40):mul(100)
--gnuplot.plot(x1, y1, '+')

-- scatter plots
--Plot = require ("itorch.Plot")
--local plot = Plot()
--plot:circle(x1,y2,'green','Group1')
--plot:circle(x2,y2,'red'  ,'Group2')
--plot:draw() 

--x = torch.zeros(4,4)
--x[1] = 1
--x[2] = -1
--gnuplot.imagesc(x)

--x=torch.randn(3,2)
--y=torch.randn(3,3)
--print(nn.JoinTable(1,1):forward{x, y})
--
----local file = 'mnist.t7/train_offcenter_100x100_10.t7'
----local f = torch.load(file)
--local file = 'mnist.t7/train_32x32.t7'
--local f = torch.load(file, 'ascii')
--local data = f.data:type(torch.getdefaulttensortype())
----gnuplot.raw("set multiplot")
------gnuplot.axis('fill')
------gnuplot.raw('set tmargin 0 \n set bmargin 0 \n set lmargin 3 \n set rmargin 3')
----gnuplot.imagesc(data[1][1])
----local x = torch.Tensor{5, 8, 14, 25}
----local y = torch.Tensor{9, 3, 15, 20}
----gnuplot.plot(x, y, '+-')
--
----local s = image.scale(data[{{1,2},1}], 16)
----gnuplot.imagesc(s[2])
--
--local L = 32
--local P, r, nScales = 8, 1/L, 3
--local l_m = torch.Tensor{{-4, 4}}
--local l_ms = torch.Tensor{{{-4,-4}}, {{-4,4}}, {{4,4}}, {{4,-4}}}
--
----gnuplot.raw('set multiplot layout 2, 2')
--local t = {}
--for i=1, 4 do
--  table.insert(t, glimpse.computePatch(data[{{1,1}}],l_ms[i],P,r)[1]:view(P,P):clone())
----  gnuplot.imagesc(t[i])
--end
--
--local reconstructed = torch.Tensor(L,L):zero()
--local all = torch.cat(torch.cat(t[1], t[2],2), torch.cat(t[4], t[3], 2), 1)
--reconstructed[{{9,24}, {9,24}}] = all
--gnuplot.imagesc(reconstructed)

--local patches = glimpse.computePatchMultipleScale1(data[{{1,1}}],l_m, P,r,nScales):view(1,3,P,P)

--local step = torch.Tensor{8, 16}
--local regular = glimpse.computePatch(data[{{1,1}}],l_m,P,r)
------local patches = glimpse.computePatchMultipleScale(data[{{1,1}}],l_m,P,r,step):view(1,3,P,P)
--gnuplot.raw('set multiplot layout 1, 2')
--gnuplot.imagesc(data[1][1])
--gnuplot.imagesc(regular[1]:view(P, P))
--gnuplot.imagesc(downsampled[1]:view(P, P))
--gnuplot.imagesc(patches[1][1])
--gnuplot.imagesc(patches[1][2])
--gnuplot.imagesc(patches[1][3])

--local all = patches.new():resize(32,32):fill(0)
--all:sub(32/2-8/2+1, 32/2+8/2, 32/2-8/2+1, 32/2+8/2):add(patches[1][1])
--all:sub(32/2-16/2+1, 32/2+16/2, 32/2-16/2+1, 32/2+16/2):add(image.scale(patches[1][2], 16))
--all:add(image.scale(patches[1][3], 32))
--gnuplot.imagesc(all)

--local transform = require 'util.DataTransformUtils'
--transform.offCenter(100)
--
--local P, H, N, O = 3, 4, 5, 6
--local net = glimpse.createNet(P,H,O,0.5)
--local x = torch.rand(N, P*P)
--local l = torch.rand(N, 2)
--net:forward({x, l})
--
--local grad_o = torch.rand(N, O)*0.1
--local grad_x_l = net:backward({x, l}, grad_o)
--print(#grad_x_l)
 
--h1 = nn.Linear(20, 20)()
--h2 = nn.Linear(10, 10)()
--hh1 = nn.Linear(20, 1)(nn.Tanh()(h1))
--hh2 = nn.Linear(10, 1)(nn.Tanh()(h2))
--madd = nn.CAddTable()({hh1, hh2})
--oA = nn.Sigmoid()(madd)
----oB = nn.Tanh()(madd)
--gmod = nn.gModule({h1, h2}, {oA})
--
--cloned = gmod:clone('weight', 'bias', 'gradWeight', 'gradBias')
--print('here')