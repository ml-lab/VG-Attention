require 'nn'
require 'torch'
require 'optim'
require 'DataLoaderDisk'
require 'misc.optim_updates'
local utils = require 'misc.utils'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a VQA simple MLP model for VG')
cmd:text()
cmd:text('Options')

-- Global image features
cmd:option('-input_img_train_h5','regional_feature/VG_img_residule_semantic_train.h5','path to the h5file containing the global image feature')
cmd:option('-input_img_val_h5','regional_feature/VG_img_residule_semantic_val.h5','path to the h5file containing the global image feature')
cmd:option('-input_img_test_h5','regional_feature/VG_img_residule_semantic_test.h5','path to the h5file containing the global image feature')

cmd:option('-feature_type', 'Residual', 'VGG or Residual')
cmd:option('-model_type', 'MLP', 'Bilinear or MLP')
cmd:option('-fast_eval', 10, 'skip_eval')

-- QA data
cmd:option('-input_json','decoys/VG_decoys_tokenized.region_augmented.json','path to the json file containing additional info and vocab')

-- Models
cmd:option('-start_from', '', 'path to a protos.model checkpoint to initialize protos.model weights from. Empty = don\'t')
cmd:option('-wordtype', 'W2V')
cmd:option('-prefix', 'mlp_VG_iqa_WUPS')
cmd:option('-batchnorm', 1)
cmd:option('-num_MC', 4, 'number of multiple choice answers for each question. Default is 4')

cmd:option('-weight_tune', 8192, 'for MLP, the dimension of W1 and W2, which must be tuned.')
cmd:option('-batch_size',100,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-image_feat_size', 2048, 'or 4096 for VGG')
cmd:option('-ques_feat_size', 300)
cmd:option('-mode', 3, 'training answer set')

-- Optimizer setting
cmd:option('-optim','sgdmom','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-base_lr', 1e-2,'learning rate') --to tune
cmd:option('-decay_type', 'step', '')
cmd:option('-gamma', 0.1, 'lr decay param')
cmd:option('-momentum', 0.9, 'momentum param')
cmd:option('-weight_decay', 1e-5, 'weight decay')
cmd:option('-step', 200000, 'lr decay param')
cmd:option('-learning_rate_decay_every', 200, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-optim_alpha',0.99,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.995,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', 800000, 'max number of iterations to run for (-1 = run forever)')

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 50000, 'how often to save a protos.model checkpoint?')
cmd:option('-checkpoint_path', 'save2/train_Harry_X/mod', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 1000, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

-- Misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'cudnn')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then
  require 'cudnn'
  end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Get Learning Rate Utils
-------------------------------------------------------------------------------

function getLearningRate(opt, iter)
  local base_lr = opt.base_lr
  if opt.decay_type == 'fixed' then
    return base_lr
  elseif opt.decay_type == 'step' then
    return base_lr * math.pow(opt.gamma, math.floor(iter / opt.step))
  elseif opt.decay_type == 'grads' then
    return base_lr * math.pow(opt.gamma, math.floor(iter / opt.step))
  elseif opt.decay == 'exp' then
    return base_lr * math.pow(opt.gamma, iter)
  end
end

local save_path = opt.checkpoint_path .. '_' .. opt.prefix .. '_' .. opt.model_type .. '_' .. opt.feature_type .. '_' .. opt.weight_tune .. '_' .. opt.optim .. '_' .. opt.base_lr .. '_' .. opt.step .. '_' .. opt.batch_size .. '_' .. opt.mode
paths.mkdir(save_path)
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_val = opt.input_img_val_h5, h5_img_file_test = opt.input_img_test_h5, json_file = opt.input_json, feature_type = opt.feature_type}
------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the protos.model...')
-- intialize language protos.model
local loaded_checkpoint
local lmOpt

if string.len(opt.start_from) > 0 then
  loaded_checkpoint = torch.load(opt.start_from)
  lmOpt = loaded_checkpoint.lmOpt
  protos = loaded_checkpoint.protos
else
  lmOpt = {}
  lmOpt.dropout = 0.5
  lmOpt.image_feat_size = opt.image_feat_size
  lmOpt.ques_feat_size = opt.ques_feat_size

  print('Building and initialzing protos.model...')

  -- MLP
  protos.model = nn.Sequential()
  protos.model:add(nn.Linear(opt.image_feat_size + 600, opt.weight_tune, false))
  if opt.batchnorm == 1 then
    protos.model:add(nn.BatchNormalization(opt.weight_tune))
  end
  protos.model:add(nn.ReLU())
  protos.model:add(nn.Dropout(lmOpt.dropout))
  protos.model:add(nn.Linear(opt.weight_tune, 1, true))
  protos.model:add(nn.Sigmoid())
  protos.crit = nn.BCECriterion()

  -- Normalization
  protos.norm_ans  = nn.Normalize(2)
  protos.norm_img  = nn.Normalize(2)
  protos.norm_que = nn.Normalize(2)
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local params, grad_params = protos.model:getParameters()
print('total number of parameters of model: ', params:nElement())
collectgarbage()

-------------------------------------------------------
-- answer to word vector -- currently no normalization
-------------------------------------------------------

print("w2vutils loading..")
local w2vutils = require 'word2vec.w2vutils'

local function read_answer(data, split, batch_ans_type, enable_deco)
  enable_deco = enable_deco or 2

  batch_ans_type = batch_ans_type or {}
  local N_batch_size = 0

  local _num_MC = opt.num_MC
  if enable_deco == 3 then
    _num_MC = 7
  elseif enable_deco == 5 then
    _num_MC = 10
  end

  N_batch_size = (opt.batch_size * _num_MC)

  local cur_ans_type = torch.zeros(opt.batch_size)
  local cur_type_count = torch.zeros(6)

  local avg_word_embedding_a = torch.zeros(N_batch_size, 300)
  local correctness_label = torch.zeros(N_batch_size)

  local answer_index = 1
  -- which of the four answers is this answer? assumes that batches are divided up evenly into groups of 4

  for IQA_index, IQA_pair in pairs(data) do
    local answer = IQA_pair[1]

    for word_index, word in pairs(answer) do
      if (word_index ~= #answer or #answer == 1) then
        avg_word_embedding_a[IQA_index]:add(w2vutils:word2vec(word))
      end
    end

    if (#answer - 1) ~= 0 then
      avg_word_embedding_a[IQA_index]:div(#answer - 1)
    end

    answer_index = answer_index + 1
    if answer_index == (_num_MC + 1) then
      answer_index = 1
      correctness_label[IQA_index] = 1
    end

  end

  if split == -1000 then
    for _ind, ans_type in pairs(batch_ans_type) do
      if ans_type == 'what' then
        cur_ans_type[_ind] = 1
        cur_type_count[1] = cur_type_count[1] + 1
      elseif ans_type == 'where' then
        cur_ans_type[_ind] = 2
        cur_type_count[2] = cur_type_count[2] + 1
      elseif ans_type == 'when' then
        cur_ans_type[_ind] = 3
        cur_type_count[3] = cur_type_count[3] + 1
      elseif ans_type == 'who' then
        cur_ans_type[_ind] = 4
        cur_type_count[4] = cur_type_count[4] + 1
      elseif ans_type == 'why' then
        cur_ans_type[_ind] = 5
        cur_type_count[5] = cur_type_count[5] + 1
      elseif ans_type == 'how' then
        cur_ans_type[_ind] = 6
        cur_type_count[6] = cur_type_count[6] + 1
      else
        print('error in type')
      end
    end
  end
  return avg_word_embedding_a, correctness_label, cur_ans_type, cur_type_count

end

-------------------------------------------------------
-- question to word vector -- currently no normalization
-------------------------------------------------------

local function read_question(data_ques, split, enable_deco)
  enable_deco = enable_deco or 2
  local N_batch_size = 0
  local _num_MC = opt.num_MC

  if enable_deco == 3 then
    _num_MC = 7
  elseif enable_deco == 5 then
    _num_MC = 10
  end
  N_batch_size = opt.batch_size

  local avg_word_embedding_q = torch.zeros(N_batch_size, 300)

  ------------
  -- Embedding
  ------------

  for IQA_index, question in pairs(data_ques) do

    for word_index, word in pairs(question) do
      if (word_index ~= #question or #question == 1) then
        avg_word_embedding_q[IQA_index]:add(w2vutils:word2vec(word))
      end
    end

    if (#question - 1)~=0 then
      avg_word_embedding_q[IQA_index]:div(#question- 1)
    end
  end

  return avg_word_embedding_q
end

-------------------------------------------------------------------------------
-- Validation evaluation -- Harry to finish
-------------------------------------------------------------------------------
local function eval_split(split, enable_deco)
  enable_deco = enable_deco or 2
  local _num_MC = opt.num_MC
  if enable_deco == 3 then
    _num_MC = 7
  elseif enable_deco == 5 then
    _num_MC = 10
  end

  -- initialize
  protos.model:evaluate()
  loader:resetIterator(split)

  local n_eval = 0
  local total_qnum = math.floor(loader:getDataNum(split) / opt.fast_eval) -- only 20 images, but 4 examples/image = batch size 80

  local loss_sum = 0
  local loss_evals = 0
  local good_answers = 0
  local total_type_count = torch.zeros(6)
  local correct_type_count = torch.zeros(6)
  local inference_data = torch.zeros(total_qnum)

 while true do
    local data, IQA_batch = loader:getBatch{batch_size = opt.batch_size, split = split, num_MC = _num_MC, deco = enable_deco, fast_rate = opt.fast_eval}
    -- read question
    local avg_word_embed_q = read_question(data.questions, split, enable_deco)
    -- read answer
    local avg_word_embed_a, correctness_label, cur_ans_type, cur_type_count = read_answer(IQA_batch, split, data.ans_type, enable_deco)
    if split == -1000 then total_type_count:add(cur_type_count) end

    if opt.gpuid >= 0 then
      data.images = data.images:cuda()
      avg_word_embed_a = avg_word_embed_a:cuda()
      avg_word_embed_q = avg_word_embed_q:cuda()
      correctness_label = correctness_label:cuda()
    end

    local img_norm = protos.norm_img:forward(data.images)
    local que_norm   = protos.norm_que:forward(avg_word_embed_q)
    local ans_norm   = protos.norm_ans:forward(avg_word_embed_a)
    local img_norm_rep = torch.repeatTensor(img_norm, 1, _num_MC):view(-1, opt.image_feat_size):contiguous()
    local que_norm_rep = torch.repeatTensor(que_norm, 1, _num_MC):view(-1, opt.ques_feat_size):contiguous()
    local input_feat_MLP = torch.cat({img_norm_rep, que_norm_rep, ans_norm}, 2)

    local out_score = protos.model:forward(input_feat_MLP)
    local loss = protos.crit:forward(out_score, correctness_label)
    local out_total_score = torch.reshape(out_score, opt.batch_size, _num_MC)
    local tmp, pred = torch.max(out_total_score, 2)

    -- batch start point in terms of question
    local batch_q_start = n_eval * opt.batch_size
    local num_ques      = math.min( total_qnum - batch_q_start, pred:size(1) )

    -- cute progress bar
    n_eval = n_eval + 1

    xlua.progress(batch_q_start + num_ques, total_qnum)
    for i = 1, num_ques do
      -- record inference data
      inference_data[batch_q_start + i] = pred[i][1]
      if pred[i][1] == _num_MC then
        good_answers = good_answers + 1
        if split == -1000 then
          correct_type_count[cur_ans_type[i]] = correct_type_count[cur_ans_type[i]] + 1
        end
      end
    end

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    if ( batch_q_start + num_ques ) >= total_qnum then break end
  end

  if split == -1000 then
    for i = 1,6 do
      correct_type_count[i] = correct_type_count[i] * 100 / total_type_count[i]
    end

    print(string.format('[acc] what: %.2f, where: %.2f, when: %.2f, who: %.2f, why: %.2f, how: %.2f', correct_type_count[1], correct_type_count[2], correct_type_count[3], correct_type_count[4], correct_type_count[5], correct_type_count[6]) )
  end

  return loss_sum/loss_evals, good_answers/total_qnum
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------

local function lossFun(enable_deco)
  enable_deco = enable_deco or 2
  local _num_MC = opt.num_MC
  if enable_deco == 3 then
    _num_MC = 7
  elseif enable_deco == 5 then
    _num_MC = 10
  end

  protos.model:training()
  grad_params:zero()

  local data, IQA_batch = loader:getBatch{batch_size = opt.batch_size, split = 0, num_MC = _num_MC, deco = enable_deco}


  local avg_word_embed_a, correctness_label = read_answer(IQA_batch, 0)
  local avg_word_embed_q = read_question(data.questions, 0)

  if opt.gpuid >= 0 then
    data.images = data.images:cuda()
    avg_word_embed_a = avg_word_embed_a:cuda()
    avg_word_embed_q = avg_word_embed_q:cuda()
    correctness_label = correctness_label:cuda()
  end

  local img_norm = protos.norm_img:forward(data.images)
  local que_norm   = protos.norm_que:forward(avg_word_embed_q)
  local ans_norm   = protos.norm_ans:forward(avg_word_embed_a)
  local img_norm_rep = torch.repeatTensor(img_norm, 1, _num_MC):view(-1, opt.image_feat_size):contiguous()
  local que_norm_rep = torch.repeatTensor(que_norm, 1, _num_MC):view(-1, opt.ques_feat_size):contiguous()
  local input_feat_MLP = torch.cat({img_norm_rep, que_norm_rep, ans_norm}, 2)

  local out_score = protos.model:forward(input_feat_MLP)
  local loss = protos.crit:forward(out_score, correctness_label)
  local dlogprobs = protos.crit:backward(out_score, correctness_label)
  local d_score = protos.model:backward(input_feat_MLP, dlogprobs)

  -- Add a weight decay
  grad_params:add(opt.weight_decay, params)

  return loss
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local n = 0
local iter = 0
local ave_loss = 0
local loss_history = {}
local accuracy_history = {} --not actually accuracy history
local val_loss_history = {}
local val_accu_history = {}
local learning_rate_history = {}
local optim_state = {}
local timer = torch.Timer()
local learning_rate = opt.base_lr

while true do

  local losses = 0
  if opt.mode == 4 then
    losses = lossFun(4)
  elseif opt.mode == 1 then
    losses = lossFun(1)
  elseif opt.mode == 2 then
    losses = lossFun(2)
  elseif opt.mode == 3 then
    local random_choice = torch.uniform()
    if random_choice <= 0.50 then
      losses = lossFun(1)
    else
      losses = lossFun(2)
    end
  elseif opt.mode == 5 then
    local random_choice = torch.uniform()
    if random_choice <= 0.45 then
      losses = lossFun(1)
    elseif random_choice <= 0.9 then
      losses = lossFun(2)
    else
      losses = lossFun(4)
    end
  elseif opt.mode == 20 then
    losses = lossFun(20)
  elseif opt.mode == 21 then
    losses = lossFun(21)
  else
    print('Error: unknown training mode')
  end
  ave_loss = ave_loss + losses

  learning_rate = getLearningRate(opt, iter)

  if iter % opt.losses_log_every == 0 then
    ave_loss = ave_loss / opt.losses_log_every
    loss_history[iter] = losses
    accuracy_history[iter] = ave_loss
    learning_rate_history[iter] = learning_rate

    print(string.format('iter %d: %f, %f, %f, %f', iter, losses, ave_loss, learning_rate, timer:time().real))

    ave_loss = 0
  end

  if ( (iter % opt.save_checkpoint_every == 0) or iter == opt.max_iters) then
      local val_loss = 0
      local val_accu = 0
      if opt.mode == 20 then
        val_loss, val_accu = eval_split(1, 3)
      elseif opt.mode == 21 then
        val_loss, val_accu = eval_split(1, 5)
      else
        val_loss, val_accu = eval_split(1, opt.mode)
      end
      print('validation loss: ', val_loss, 'accuracy ', val_accu)
      val_loss_history[iter] = val_loss
      val_accu_history[iter] = val_accu

      local test_loss = 0
      local test_accu = 0
      if opt.mode == 20 then
        test_loss, test_accu = eval_split(2, 3)
      elseif opt.mode == 21 then
        test_loss, test_accu = eval_split(2, 5)
      else
        test_loss, test_accu = eval_split(2, opt.mode)
      end
      print('test loss: ', test_loss, 'accuracy ', test_accu)

      local checkpoint_path = path.join(save_path, 'model_id' .. opt.id .. '_iter'.. iter)
      print('wrote model to ' .. checkpoint_path .. '.')
      torch.save(checkpoint_path..'.t7', {protos=protos, lmOpt=lmOpt})

      local checkpoint = {}
      checkpoint.val_loss_history = val_loss_history
      checkpoint.val_accu_history = val_accu_history
      checkpoint.opt = opt
      checkpoint.iter = iter
      checkpoint.loss_history = loss_history
      checkpoint.accuracy_history = accuracy_history
      checkpoint.learning_rate_history = learning_rate_history

      local checkpoint_path = path.join(save_path, 'checkpoint.json')
      utils.write_json(checkpoint_path, checkpoint)
      print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

  end

  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgdmom' then
    -- sgd with momentum setted as 0.9
    sgdmom(params, grad_params, learning_rate, opt.momentum, optim_state)
  else
    error('bad option opt.optim')
  end

  iter = iter + 1
  if opt.max_iters > 0 and iter > opt.max_iters then break end -- stopping protos.criterion
end
