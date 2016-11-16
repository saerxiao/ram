require 'torch'
require 'nngraph'
require 'gnuplot'


--local file = 'mnist.t7/train_offcenter_100x100_10.t7'
local file = 'mnist.t7/train_32x32.t7'
local f = torch.load(file)
local data = f.data:type(torch.getdefaulttensortype())
gnuplot.imagesc(data[1][1])

--local transform = require 'util.DataTransformUtils'
--transform.offCenter(100)

--local glimpse = require 'Glimpse'
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