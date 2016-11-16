require 'torch'
require 'paths'

local Loader = {}
Loader.__index = Loader

local function loadDataset(fileName, maxLoad)
    local f = nil
    if string.match(fileName, '32x32') then
      f = torch.load(fileName, 'ascii')
    else
      f = torch.load(fileName)
    end
    local data = f.data:type(torch.getdefaulttensortype())
    local labels = f.labels

    local nExample = f.data:size(1)
    if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
    end
    data = data[{{1,nExample},{},{},{}}]
    labels = labels[{{1,nExample}}]

    local dataset = {}
    dataset.data = data
    dataset.labels = labels

    function dataset:size()
      return nExample
    end

    return dataset
end

-- attributes in opt, all optional
-- batch_size, default 1
-- validate_split_fraction, defualt 0.9
-- train_max_load, default all
-- test_max_load, default all
function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)

  local path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
  local path_dataset = 'mnist.t7'
  local path_trainset = paths.concat(path_dataset, 'train_32x32.t7')
  local path_testset = paths.concat(path_dataset, 'test_32x32.t7')
  if not paths.filep(path_trainset) or not paths.filep(path_testset) then
      local remote = path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
  end

  self.trainingData = loadDataset(path_trainset, opt.train_max_load)
  
  
  self.training_size = math.ceil(self.trainingData.size()* (opt.validate_split or 0.9))

  self.batch_size = opt.batch_size or 1
  local batches = {}
  batches.train_data = self.trainingData.data[{{1, self.training_size}}]:split(self.batch_size)
  batches.train_labels = self.trainingData.labels[{{1, self.training_size}}]:split(self.batch_size)
  batches.validate_data = self.trainingData.data[{{self.training_size+1, self.trainingData:size()}}]:split(self.batch_size)
  batches.validate_labels = self.trainingData.labels[{{self.training_size+1, self.trainingData:size()}}]:split(self.batch_size)
  
  self.testset = loadDataset(path_testset, opt.test_max_load)
  batches.test_data = self.testset.data:split(self.batch_size)
  batches.test_labels = self.testset.labels:split(self.batch_size)
  self.batches = batches
  
  local split_size = {}
  split_size.train = #batches.train_data
  split_size.validate = #batches.validate_data
  split_size.test = #batches.test_data
  self.split_size = split_size

  return self
end

function Loader:get_train_data()
  return self.trainingData.data[{{1, self.training_size}}], self.trainingData.labels[{{1, self.training_size}}]
end

function Loader:get_validate_data()
  return self.trainingData.data[{{self.training_size+1, self.trainingData:size()}}],
         self.trainingData.labels[{{self.training_size+1, self.trainingData:size()}}]
end

function Loader:iterator(split)
  local it = {}
  local cursor = 0
  it.reset = function()
    cursor = 0
  end
  it.next_batch = function()
    local data, labels = self.batches[split .. "_data"], self.batches[split .. "_labels"]
    cursor = cursor + 1
    if cursor > #data then
      cursor = 1
    end
    return data[cursor], labels[cursor]
  end
  return it
end

return Loader
