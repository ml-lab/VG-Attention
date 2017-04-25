require 'nn'
require 'optim'
require 'torch'
require 'math'
require 'cunn'
require 'cutorch'
require 'image'
require 'hdf5'
cjson=require('cjson')
require 'xlua'
require 'cudnn'
local t = require 'transforms'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','augmented_decoys/VG_val_decoys.region_augmented.json','path to the json file containing vocab and answers')
cmd:option('-image_root','data/image/','path to the image root')

cmd:option('-residule_path', 'image_model/resnet-200.t7')
cmd:option('-batch_size', 5, 'batch_size')
cmd:option('-mode', 2, 'mode')

cmd:option('-out_name', 'regional_feature/vg_img_residule_attended_val.h5', 'output name train')

cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid+1)
local model = torch.load(opt.residule_path)


for i = 14,12,-1 do
  model:remove(i)
end
print(model)
model:evaluate()
model=model:cuda()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.ColorNormalize(meanstd)
}

imloader={}
function imloader:load(fname)
    self.im=image.load(fname)
end

function loadim(imname)
  imloader:load(imname,  3, 'float')
  im = imloader.im
  im = image.scale(im, 448, 448)

  if im:size(1)==1 then
    im2=torch.cat(im,im,1)
    im2=torch.cat(im2,im,1)
    im=im2
  elseif im:size(1)==4 then
    im=im[{{1,3},{},{}}]
  end

  im = transform(im)
  return im
end

function obtain_mask(imname, region)
  imloader:load(imname, 3, 'float')
  im = imloader.im

  local hfactor = ( 13 / im:size(2) )
  local wfactor = ( 13 / im:size(3) )

  local x1 = torch.round( region['x1'] * wfactor ) + 1
  local x2 = torch.round( region['x2'] * wfactor ) + 1
  local y1 = torch.round( region['y1'] * hfactor ) + 1
  local y2 = torch.round( region['y2'] * hfactor ) + 1
  -- print(string.format('x1: %d, x2: %d, y1: %d, y2: %d', region['x1'], region['x2'], region['y1'], region['y2']))
  -- print(string.format('x1: %d, x2: %d, y1: %d, y2: %d', x1, x2, y1, y2))

  local att = torch.zeros(14, 14)
  att[{{x1,x2}, {y1,y2}}]:fill(1)
  return att
end

local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local img_list, region_list={}, {}
for i,image_item in pairs(json_file) do
  table.insert(img_list, image_root .. image_item['image_id'] .. '.jpg')
  table.insert(region_list, { x1 = image_item['x'], x2 = image_item['x'] + image_item['width'], y1 = image_item['y'], y2 = image_item['y'] + image_item['height'] })
end


local ndims=2048
local batch_size = opt.batch_size
local sz=#img_list
local feat=torch.FloatTensor(sz,2048) --ndims)

local view_attention=nn.View(batch_size, 196, 1):cuda()
local view_feature=nn.View(batch_size, 196, 2048):cuda()
local attend=nn.MM(true, false):cuda()
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,448,448)
    local masks=torch.zeros(batch_size, 14, 14)
    for j=1,r-i+1 do
      ims[j]=loadim(img_list[i+j-1]):cuda()
      masks[j]=obtain_mask(img_list[i+j-1],region_list[i+j-1])
    end
    local output=model:forward(ims)
    -- print('- output feature shape: ' .. tostring(output:size()) )
    -- print('- attention shape: ' .. tostring(masks:size()) )

    local img_feature = view_feature:forward(output:cuda())
    local attention   = view_attention:forward(masks:cuda())
    local attended_feature = attend:forward({img_feature, attention})

    -- print('- attended feature shape: ' .. tostring(attended_feature:size()))

    feat[{{i,r},{}}]=attended_feature:contiguous():float()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images', feat)
train_h5_file:write('/attention', att)
train_h5_file:close()
