#!/usr/bin/env th
-- 
-- An Analysis of Single-Layer Networks in Unsupervised Feature Learning
-- by Adam Coates et al. 2011
--
-- The original MatLab code can be found in http://www.cs.stanford.edu/~acoates/
-- Tranlated to Lua/Torch7
--
require("xlua")
require("image")
require("unsup")
require("kmeans")
require("extract")
require("train-svm")
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
opt = {
   whiten = true,
}

-- set parameters
local CIFAR_dim = {3, 32, 32}
local trsize = 50000
local tesize = 10000
local kSize = 7
local nkernel1 = 32
local nkernel2 = 32
local fanIn1 = 1
local fanIn2 = 4


print("==> download dataset")
if not paths.dirp('cifar-10-batches-t7') then
   tar = 'http://data.neuflow.org/data/cifar10.t7.tgz'
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end


print("==> load dataset")
local trainData = {
   data = torch.Tensor(50000, CIFAR_dim[1]*CIFAR_dim[2]*CIFAR_dim[3]),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0, 4 do
   local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]
trainData.data = trainData.data:reshape(trsize,CIFAR_dim[1],CIFAR_dim[2],CIFAR_dim[3])

local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
local testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

testData.data   = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]
testData.data = testData.data:reshape(tesize,3,32,32)


print("==> extract patches")
local numPatches = 50000
local patches = torch.zeros(numPatches, kSize*kSize*CIFAR_dim[1])
for i = 1,numPatches do
   xlua.progress(i,numPatches)
   local r = torch.random(CIFAR_dim[2] - kSize + 1)
   local c = torch.random(CIFAR_dim[3] - kSize + 1)
   patches[i] = trainData.data[{math.fmod(i-1,trsize)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
   patches[i] = patches[i]:add(-patches[i]:mean())
   patches[i] = patches[i]:div(math.sqrt(patches[i]:var()+10))
end


if opt.whiten then
   print("==> whiten patches")
   local function zca_whiten(x)
      local dims = x:size()
      local nsamples = dims[1]
      local ndims    = dims[2]
      local M = torch.mean(x, 1)
      local D, V = unsup.pcacov(x)
      x:add(torch.ger(torch.ones(nsamples), M:squeeze()):mul(-1))
      local diag = torch.diag(D:add(0.1):sqrt():pow(-1))
      local P = V * diag * V:t()
      x = x * P
      return x, M, P
   end
   patches, M, P = zca_whiten(patches)
end


print("==> find clusters")
local ncentroids = 1600
kernels, counts = unsup.kmeans_modified(patches, ncentroids, nil, 0.1, 1, 1000, nil, true)


print("==> select distinct features")
local j = 0
for i = 1,ncentroids do
   if counts[i] > 0 then
      j = j + 1
      kernels[{j,{}}] = kernels[{i,{}}]
      counts[j] = counts[i]
   end
end
kernels = kernels[{{1,j},{}}]
counts  = counts[{{1,j}}]
-- just select 1600 kernels for now


print("==> feed-forward")
local pmetric = 'max'   -- conv + pool
local trainFeatures
if opt.whiten then
   trainFeatures = extract_features(trainData.data, kernels, kSize, nil, pmetric, M, P)
else
   trainFeatures = extract_features(trainData.data, kernels, kSize, nil, pmetric)
end


print("==> normalize data")
for i = 1,trsize do
   xlua.progress(i,trsize)
   local trmean = trainFeatures[{i,{}}]:mean()
   local trstd  = math.sqrt(trainFeatures[{i,{}}]:var()+0.01)
   trainFeatures[{i,{}}] = trainFeatures[{i,{}}]:add(-trmean):div(trstd)
end


print("==> train SVM classifier")
trainFeatures = torch.cat(trainFeatures, torch.ones(trainFeatures:size(1)), 2)
local theta = train_svm(trainFeatures, trainData.labels, 100);
local val,idx = torch.max(trainFeatures * theta, 2)
local match = torch.eq(trainData.labels, idx:float():squeeze()):sum()
local accuracy = match/trainData.labels:size(1)*100
print('==> train accuracy is '..accuracy..'%')



print("==> process testing data")
local testFeatures
if opt.whiten then
   testFeatures = extract_features(testData.data, kernels, kSize, nil, pmetric, M, P)
else
   testFeatures = extract_features(testData.data, kernels, kSize, nil, pmetric)
end


print("==> report results")
testFeatures = torch.cat(testFeatures, torch.ones(testFeatures:size(1)), 2)
local val,idx = torch.max(testFeatures * theta, 2)
local match = torch.eq(testData.labels, idx:float():squeeze()):sum()
local accuracy = match/testData.labels:size(1)*100
print("==> test accuracy is "..accuracy.."%")
