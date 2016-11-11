require 'torch'
require 'nngraph'

h1 = nn.Linear(20, 20)()
h2 = nn.Linear(10, 10)()
hh1 = nn.Linear(20, 1)(nn.Tanh()(h1))
hh2 = nn.Linear(10, 1)(nn.Tanh()(h2))
madd = nn.CAddTable()({hh1, hh2})
oA = nn.Sigmoid()(madd)
--oB = nn.Tanh()(madd)
gmod = nn.gModule({h1, h2}, {oA})

cloned = gmod:clone('weight', 'bias', 'gradWeight', 'gradBias')
print('here')