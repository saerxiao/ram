require 'torch'
require 'paths'

local Loader = {}
Loader.__index = Loader

local function loadDataset(fileName, maxLoad)
    local f = torch.load(fileName, 'ascii')
    local data = f.data:type(torch.getdefaulttensortype())
    local labels = f.labels

    local nExample = f.data:size(1)
    if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
    end
    data = data[{{1,nExample},{},{},{}}]
    labels = labels[{{1,nExample}}]
    
--    if not isConvNet then
--      local size_per_row = 1
--      for i = 2,data:size():size() do
--        size_per_row = size_per_row * data:size(i)
--      end
--      data = data:view(data:size(1), size_per_row)
--    end

    local dataset = {}
    dataset.data = data
    dataset.labels = labels
    
--    function dataset:normalize(mean_, std_)
--      local mean = mean_ or data:view(data:size(1), -1):mean(1)
--      local std = std_ or data:view(data:size(1), -1):std(1, true)
--      for i=1,data:size(1) do
--         data[i]:add(-mean[1][i])
--         if std[1][i] > 0 then
--            tensor:select(2, i):mul(1/std[1][i])
--         end
--      end
--      return mean, std
--    end
--
--    function dataset:normalizeGlobal(mean_, std_)
--      local std = std_ or data:std()
--      local mean = mean_ or data:mean()
--      data:add(-mean)
--      data:mul(1/std)
--      return mean, std
--    end

    function dataset:size()
      return nExample
    end

--    local labelstensor = torch.zeros(nExample, 10)
--    for i = 1, nExample do
--      labelstensor[i][labels[i]] = 1
--    end
--    dataset.labels = labelstensor
--    local labelvector = torch.zeros(10)
--
--    setmetatable(dataset, {__index = function(self, index)
--           local input = self.data[index]
--           local class = self.labels[index]
--           local label = labelvector:zero()
--           label[class] = 1
--           local example = {input, label}
--                                       return example
--    end})

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
--  trainset = get_subset(trainingData, 1, training_size)
--  self.valset = get_subset(trainingData,training_size+1, trainingData:size(1))
  
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
  
--  local batch_cursor = {}
--  batch_cursor.train = 0
--  batch_cursor.validate = 0
--  batch_cursor.test = 0
--  self.batch_cursor = batch_cursor

  return self
end

function Loader:get_train_data()
  return self.trainingData.data[{{1, self.training_size}}], self.trainingData.labels[{{1, self.training_size}}]
end

function Loader:get_validate_data()
  return self.trainingData.data[{{self.training_size+1, self.trainingData:size()}}],
         self.trainingData.labels[{{self.training_size+1, self.trainingData:size()}}]
end

--function Loader:reset_batch(split)
--  self.batch_cursor[split] = 0
--end

--function Loader:next_batch(split)
--  self.batch_cursor[split] = self.batch_cursor[split] + 1
--  if (self.batch_cursor[split] > #self.batches[split .. "_data"]) then
--    self.batch_cursor[split] = 1
--  end
--  return self.batches[split .. "_data"][self.batch_cursor[split]],
--         self.batches[split .. "_labels"][self.batch_cursor[split]]
--end

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
