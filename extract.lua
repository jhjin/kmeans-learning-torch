-- 
-- An Analysis of Single-Layer Networks in Unsupervised Feature Learning
-- by Adam Coates et al. 2011
--
-- The original MatLab code can be found in http://www.cs.stanford.edu/~acoates/
-- Tranlated to Lua/Torch7
--
function extract_features(img, kernels, kSize, bias, pmetric, M, P, norm)

   -- shortcuts
   local zeros = torch.zeros
   local ones  = torch.ones
   local ger   = torch.ger

   -- input/output dimensions
   local isample  = img:size(1)
   local ichannel = img:size(2)
   local iheight  = img:size(3)
   local iwidth   = img:size(4)
   local nkernels = kernels:size(1)
   local klength  = kernels:size(2)
   local kheight  = kSize
   local kwidth   = kSize
   local cheight  = iheight - kheight + 1
   local cwidth   = iwidth  - kwidth  + 1
   local npatches = cheight*cwidth

   -- flags
   local norm_  = (norm == nil) or (norm == true)  -- data normalized by default
   local bias_  = not (bias == nil)                -- no bias if not specified
   local whiten = not (P == nil)                   -- no whiten if not specified

   -- init pooling layer
   require 'nnx'
   local pheight = torch.floor(cheight/2)
   local pwidth  = torch.floor(cwidth/2)
   if pmetric == 'max' then
      print '==> max-pooling is applied'
      pooler = nn.SpatialMaxPooling(pwidth, pheight, pwidth, pheight)
   elseif pmetric == 'sum' then
      print '==> L1-pooling is applied'
      pooler = nn.SpatialLPPooling(nkernel, 1, pwidth, pheight, pwidth, pheight)
   end

   local features = zeros(isample, nkernels*4) 
   for k = 1,isample do
      xlua.progress(k, isample)
      -- image to column vectors
      local patches = zeros(npatches, klength)
      for i = 1,cheight do
         for j = 1,cwidth do
            local ptr   = (i-1)*cwidth + j
            local patch = img[{k,{},{i,i+kheight-1},{j,j+kwidth-1}}]
            patches[{ptr,{}}] = patch:reshape(klength)
         end
      end
   
      -- local normalization
      if norm_ then
         local pmean   = ger(patches:mean(2):squeeze(),ones(klength))
         local pstd    = ger(patches:var(2):add(10):sqrt():squeeze(),ones(klength))
         patches = patches:add(pmean:mul(-1)):cdiv(pstd)
      end

      -- whiten
      if whiten then
         patches:add(ger(ones(npatches), M:squeeze()):mul(-1))
         patches = patches * P
      end

      -- extract features by convolution and bias
      local feature = patches * kernels:t()
      if bias_ then feature = feature:add(ger(ones(npatches), bias:squeeze())) end

      -- thresholding
      local threshold = 0
      feature = feature:cmul(torch.gt(feature, threshold):float())
      feature = feature:t():reshape(nkernels,cheight,cwidth)

      -- pooling
      local pooled = pooler:forward(feature)
      pooled = pooled:reshape(nkernels*pooled:size(2)*pooled:size(3))
      -- save feature vector
      features[{k,{}}] = pooled
   end

   return features
end
