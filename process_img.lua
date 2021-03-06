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
cmd:option('-input_json','augmented_decoys/VG_test_decoys.region_augmented.json','path to the json file containing vocab and answers')
cmd:option('-image_root','data/image/','path to the image root')

cmd:option('-residule_path', 'image_model/resnet-200.t7')
cmd:option('-batch_size', 40, 'batch_size')
cmd:option('-mode', 2, 'mode')

cmd:option('-out_name', 'regional_feature/vg_img_residule_semantic_test.h5', 'output name train')

cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

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
function loadim(imname, region)
    imloader:load(imname,  3, 'float')
    im = imloader.im
    im = image.crop(im, region['x1'], region['y1'], region['x2'], region['y2'])
    im = image.scale(im, 224, 224)

    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end

    return im
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
print(string.format('processing %d images...',sz))
for i=1,10 do
  im=loadim(img_list[i], region_list[i]):cuda()
  print(string.format("image id: %s, region id: %s, question: %s, phrase: %s", json_file[i]['image_id'], json_file[i]['region_id'], json_file[i]['question'], json_file[i]['phrase']))
  image.save('images/' .. json_file[i]['image_id'] .. '_' .. json_file[i]['region_id'] .. '.jpg', im)
  collectgarbage()
end
