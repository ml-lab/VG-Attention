require 'hdf5'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

    -- global image feature
    if opt.h5_img_file_train ~= nil then
        print('DataLoader loading h5 image file train: ', opt.h5_img_file_train)
        self.h5_img_file_train = hdf5.open(opt.h5_img_file_train, 'r')
    end

    if opt.h5_img_file_val ~= nil then
        print('DataLoader loading h5 image file val: ', opt.h5_img_file_val)
        self.h5_img_file_val = hdf5.open(opt.h5_img_file_val, 'r')
    end

    if opt.h5_img_file_test ~= nil then
        print('DataLoader loading h5 image file test: ', opt.h5_img_file_test)
        self.h5_img_file_test = hdf5.open(opt.h5_img_file_test, 'r')
    end

    print('DataLoader loading json file: ', opt.json_file)
    local json_file = utils.read_json(opt.json_file)
    self.feature_type = opt.feature_type

    self.ans_train_vdeco = {}
    self.ans_train_qdeco = {}
    self.ans_val_vdeco = {}
    self.ans_val_qdeco = {}
    self.ans_test_vdeco = {}
    self.ans_test_qdeco = {}
    self.type_test = {}

    self.ques_train = {}
    self.ques_val = {}
    self.ques_test = {}

    for ii,img in pairs(json_file.train) do
        table.insert(self.ans_train_vdeco, img['IoU_decoys'])
        table.insert(self.ans_train_qdeco, img['QoU_decoys'])
        table.insert(self.ques_train, img['question'])
    end

    for ii,img in pairs(json_file.val) do
        table.insert(self.ans_val_vdeco, img['IoU_decoys'])
        table.insert(self.ans_val_qdeco, img['QoU_decoys'])
        table.insert(self.ques_val, img['question'])
    end

    for ii,img in pairs(json_file.test) do
        table.insert(self.ans_test_vdeco, img['IoU_decoys'])
        table.insert(self.ans_test_qdeco, img['QoU_decoys'])
        table.insert(self.ques_test, img['question'])
        table.insert(self.type_test, img['type'])
    end

    -- Let's get the split for train and val and test.
    self.split_ix = {}
    self.iterators = {}

    for i = 1,#self.ques_train do
      local idx = 0
      if not self.split_ix[idx] then
          self.split_ix[idx] = {}
          self.iterators[idx] = 1
      end
      table.insert(self.split_ix[idx], i)
    end

    for i = 1,#self.ques_val do
      local idx = 1
      if not self.split_ix[idx] then
          self.split_ix[idx] = {}
          self.iterators[idx] = 1
      end
      table.insert(self.split_ix[idx], i)
    end

    for i = 1,#self.ques_test do
      local idx = 2
      if not self.split_ix[idx] then
          self.split_ix[idx] = {}
          self.iterators[idx] = 1
      end
      table.insert(self.split_ix[idx], i)
    end

    for k,v in pairs(self.split_ix) do
        print(string.format('assigned %d images to split %s', #v, k))
    end
    collectgarbage() -- do it often and there is no harm ;)
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getDataNum(split)
    return #self.split_ix[split]
end

function DataLoader:getBatch(opt)
    local batch_size = utils.getopt(opt, 'batch_size') 
    local split = utils.getopt(opt, 'split')
    local num_MC = utils.getopt(opt, 'num_MC')
    local enable_deco = utils.getopt(opt, 'deco', 2)
    local fast_rate = utils.getopt(opt, 'fast_rate', 1)

    if enable_deco == 3 then
      num_MC = 7
    elseif enable_deco == 5 then
      num_MC = 10
    end

    local split_ix = self.split_ix[split]
    assert(split_ix, 'split ' .. split .. ' not found.')

    local max_index = #split_ix
    local infos = {}
    local ques_idx = torch.LongTensor(batch_size):fill(0)
    local img_idx = torch.LongTensor(batch_size):fill(0)

    if self.feature_type == 'VGG' then
        self.img_batch = torch.Tensor(batch_size, 4096)
    elseif self.feature_type == 'Residual' then
        self.img_batch = torch.Tensor(batch_size, 2048)
    end

    for i=1,batch_size do
      local ri = self.iterators[split] -- get next index from iterator
      local ri_next = ri + math.floor(fast_rate) -- 1 -- increment iterator ****
      if ri_next > max_index then ri_next = 1 end
      self.iterators[split] = ri_next
      if split == 0 then
          ix = split_ix[torch.random(max_index)]
      else
          ix = split_ix[ri]
      end
      assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
      ques_idx[i] = ix
      if split == 0 then -- or split == 1 then
        -- global
        if self.h5_img_file_train ~= nil then
          local img = self.h5_img_file_train:read('/images'):partial({ix,ix},{1,2048})
          self.img_batch[i] = img
        end
      elseif split == 1 then
        -- global
        if self.h5_img_file_val ~= nil then
          local img = self.h5_img_file_val:read('/images'):partial({ix,ix},{1,2048})
          self.img_batch[i] = img
        end
      else
        -- global
        if self.h5_img_file_test ~= nil then
          local img = self.h5_img_file_test:read('/images'):partial({ix,ix},{1,2048})
          self.img_batch[i] = img
        end
      end
    end

    local data = {}
    -- fetch the question and image features.
    if split == 0 then-- or split == 1 then
      data.images = self.img_batch:view(batch_size, -1):contiguous()
      data.answer_qdeco = {}
      data.answer_vdeco = {}
      data.questions = {}
      for j=1,batch_size do
        table.insert(data.answer_qdeco, self.ans_train_qdeco[ques_idx[j]])
        table.insert(data.answer_vdeco, self.ans_train_vdeco[ques_idx[j]])
        table.insert(data.questions, self.ques_train[ques_idx[j]])
      end
    elseif split == 1 then
      data.images = self.img_batch:view(batch_size, -1):contiguous()
      data.answer_qdeco = {}
      data.answer_vdeco = {}
      data.questions = {}
      for j=1,batch_size do
        table.insert(data.answer_qdeco, self.ans_val_qdeco[ques_idx[j]])
        table.insert(data.answer_vdeco, self.ans_val_vdeco[ques_idx[j]])
        table.insert(data.questions, self.ques_val[ques_idx[j]])
      end
    else
      data.images = self.img_batch:view(batch_size, -1):contiguous()
      data.answer_qdeco = {}
      data.answer_vdeco = {}
      data.ans_type = {}
      data.questions = {}
      for j=1,batch_size do
        table.insert(data.answer_qdeco, self.ans_test_qdeco[ques_idx[j]])
        table.insert(data.answer_vdeco, self.ans_test_vdeco[ques_idx[j]])
        table.insert(data.ans_type, self.type_test[ques_idx[j]])
        table.insert(data.questions, self.ques_test[ques_idx[j]])
      end
    end

    -----------------------------------------------
    -- Make a triplet + whether it's correct or not
    -----------------------------------------------
    local final_batch = {} -- all IQA pairs for the batch

    if enable_deco == 5 then
      for Q_index=1,batch_size do
        for a_index=1, num_MC do
          local IQA_pair = {} -- one IQA pair
          if a_index <= 3 then
            table.insert(IQA_pair, data.answer_vdeco[Q_index][a_index])
          elseif a_index <= 6 then
            table.insert(IQA_pair, data.answer_qdeco[Q_index][a_index - 3])
          else
            table.insert(IQA_pair, data.answer_high[Q_index][a_index - 6])
          end

          if a_index == num_MC then -- last MC = correct answer (answer #4 of MC)
              table.insert(IQA_pair, 1)
          else
              table.insert(IQA_pair, 0)
          end
          table.insert(final_batch, IQA_pair)
        end
      end
    elseif enable_deco == 3 then
      for Q_index=1,batch_size do
        for a_index=1, num_MC do
          local IQA_pair = {} -- one IQA pair
          if a_index <= 3 then
            table.insert(IQA_pair, data.answer_vdeco[Q_index][a_index])
          else
            table.insert(IQA_pair, data.answer_qdeco[Q_index][a_index - 3])
          end

          if a_index == num_MC then -- last MC = correct answer (answer #4 of MC)
              table.insert(IQA_pair, 1)
          else
              table.insert(IQA_pair, 0)
          end
          table.insert(final_batch, IQA_pair)
        end
      end

    elseif enable_deco == 20 then -- IA + QA
      for Q_index=1,batch_size do
        local tmp_enable_deco = 0
        local random_choice = torch.uniform()
        if random_choice <= 0.5 then
          tmp_enable_deco = 1
        else
          tmp_enable_deco = 2
        end

        for a_index=1, num_MC do
          local IQA_pair = {} -- one IQA pair
          if tmp_enable_deco == 1 then
            table.insert(IQA_pair, data.answer_vdeco[Q_index][a_index])
          elseif tmp_enable_deco == 2 then
            table.insert(IQA_pair, data.answer_qdeco[Q_index][a_index])
          end

          if a_index == num_MC then -- last MC = correct answer (answer #4 of MC)
              table.insert(IQA_pair, 1)
          else
              table.insert(IQA_pair, 0)
          end
          table.insert(final_batch, IQA_pair)
        end
      end
    elseif enable_deco == 21 then -- All + rand
      for Q_index=1,batch_size do
        local tmp_enable_deco = 0
        local random_choice = torch.uniform()
        if random_choice <= 0.333 then
          tmp_enable_deco = 4
        elseif random_choice <= 0.666 then
          tmp_enable_deco = 1
        else
          tmp_enable_deco = 2
        end

        for a_index=1, num_MC do
          local IQA_pair = {} -- one IQA pair
          if tmp_enable_deco == 1 then
            table.insert(IQA_pair, data.answer_vdeco[Q_index][a_index])
          elseif tmp_enable_deco == 2 then
            table.insert(IQA_pair, data.answer_qdeco[Q_index][a_index])
          end

          if a_index == num_MC then -- last MC = correct answer (answer #4 of MC)
              table.insert(IQA_pair, 1)
          else
              table.insert(IQA_pair, 0)
          end
          table.insert(final_batch, IQA_pair)
        end
      end
    else
      for Q_index=1,batch_size do
        for a_index=1, num_MC do
          local IQA_pair = {} -- one IQA pair
          if enable_deco == 1 then
            table.insert(IQA_pair, data.answer_vdeco[Q_index][a_index])
          elseif enable_deco == 2 then
            table.insert(IQA_pair, data.answer_qdeco[Q_index][a_index])
          end

          if a_index == num_MC then -- last MC = correct answer (answer #4 of MC)
              table.insert(IQA_pair, 1)
          else
              table.insert(IQA_pair, 0)
          end
          table.insert(final_batch, IQA_pair)
        end
      end
    end

    return data, final_batch --final_batch
end
