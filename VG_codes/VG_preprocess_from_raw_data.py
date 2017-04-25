import json
import os
import argparse
import random

def main():
  train = []
  val = []
  test  = []
  imdir = '/mnt/DATA/weilunc/VG/img/VG_total/%s%s'

  VG = json.load(open('/mnt/DATA/weilunc/VG/question_answers.json', 'r'))
  VG_meta = json.load(open('VG_data_split.json', 'r'))
  count = 0
  print([len(VG_meta['split']), len(VG_meta['in_v7w'])])

  for img in VG:
    qa_pairs = img['qas']

    for qa in qa_pairs:
      image_id = qa['image_id']
      question_id = qa['qa_id']
      question = qa['question']
      answer = qa['answer']
      image_path = imdir%(image_id, '.jpg')
      in_v7w = VG_meta['in_v7w'][count]

      item = {'Ques_id': question_id, 'Img_path': image_path, 'Img_id': image_id, 'Ques': question, 'Ans': answer, 'In_v7w': in_v7w}

      if VG_meta['split'][count] == 0:
        train.append(item)
      elif VG_meta['split'][count] == 1:
        val.append(item)
      elif VG_meta['split'][count] == 2:
        test.append(item)
      else:
        print('wrong')
      count += 1

  print(count)
  # 1445322
  print(len(train) + len(val) + len(test))

  # print 'Training sample %d, Testing sample %d...' %(len(train), len(test))

  json.dump(train, open('VG_raw_train.json', 'w'))
  json.dump(test, open('VG_raw_test.json', 'w'))
  json.dump(val, open('VG_raw_val.json', 'w'))

if __name__ == "__main__":
  main()









