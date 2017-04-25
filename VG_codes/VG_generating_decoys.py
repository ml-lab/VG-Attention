from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import json
import re
import math


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];


def prepro_question(imgs, params):

    # preprocess all the question # (Harry: here to use the text package)
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['Ques']
        s_a = img['Ans']


        txt_mc = []
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
            txt_a = num_equalize(word_tokenize(str(s_a).lower()))
            # for j,s_mc in enumerate(img['MC_ans']):
            #     txt_mc.append(num_equalize(word_tokenize(str(s_mc).lower())))
        else:
            txt = tokenize(s)
            txt_a = num_equalize(tokenize(s_a))
            # for j,s_mc in enumerate(img['MC_ans']):
            #     txt_mc.append(num_equalize(tokenize(s_mc)))

        img['processed_tokens'] = txt
        img['processed_Ans_tokens'] = txt_a
        # img['processed_Mc_tokens'] = txt_mc

        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs


def get_unqiue_A(imgs):
    count_A = {}
    for img in imgs:
        count_A[str(img['Ans']).lower()] = count_A.get(str(img['Ans']).lower(), 0) + 1
    unique_A = [w for w, n in count_A.iteritems()]
    print(len(unique_A))
    cw = sorted([(count, w) for w, count in count_A.iteritems()], reverse=True)
    top_10_ans = []
    top_10_ans_original = []
    for i in range(100):
        if num_equalize(word_tokenize(cw[i][1])) not in top_10_ans:
            top_10_ans.append(num_equalize(word_tokenize(cw[i][1])))
            top_10_ans_original.append(cw[i][1])
        if len(top_10_ans) >= 10:
            break
    print(top_10_ans_original)

    return top_10_ans, top_10_ans_original


def add_frequent_A(imgs, top_10_ans, top_10_ans_original):
    WUPS_thres = 0.9
    dump_choice = [["nan", "."], ["inf", "."], ["minus", "inf", "."]]
    dump_choice_original = ["nan.", "inf.", "minus inf."]
    NN = len(top_10_ans)
    print(NN)
    for jj, img in enumerate(imgs):
        img['High_freq_A'] = []
        img['High_freq_A_original'] = []
        temp_order = np.random.permutation(NN)
        for i in range(NN):
            if (top_10_ans[temp_order[i]] != img['processed_Ans_tokens']) and (
                '-'.join(top_10_ans[temp_order[i]][:-1]) not in '-'.join(img['processed_Ans_tokens'][:-1])) and (
                '-'.join(img['processed_Ans_tokens'][:-1]) not in '-'.join(top_10_ans[temp_order[i]][:-1])) and (
                ''.join(top_10_ans[temp_order[i]][:-1]) not in ''.join(img['processed_Ans_tokens'][:-1])) and (
                ''.join(img['processed_Ans_tokens'][:-1]) not in ''.join(top_10_ans[temp_order[i]][:-1])):
                if comb_WUPS(img['processed_Ans_tokens'], top_10_ans[temp_order[i]]) <= WUPS_thres:
                    img['High_freq_A'].append(top_10_ans[temp_order[i]])
                    img['High_freq_A_original'].append(top_10_ans_original[temp_order[i]])
            if len(img['High_freq_A']) >= 3:
                break
        if len(img['High_freq_A']) < 3:
            print('GG!!')
            print(len(img['High_freq_A']))
            temp_length = len(img['High_freq_A'])
            print(img['processed_Ans_tokens'])
            temp_order2 = np.random.permutation(3)
            for j in range(3 - temp_length):

                img['High_freq_A'].append(dump_choice[temp_order2[j]])
                img['High_freq_A_original'].append(dump_choice_original[temp_order2[j]])


            print(len(img['High_freq_A']))
        img['High_freq_A'].append(img['processed_Ans_tokens'])
        img['High_freq_A_original'].append(img['Ans'])

        if jj % 1000 == 0:
            print('in high freq')
            print(jj)
    return imgs


def num_equalize(str_seq):
    num_list = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                '8': 'eight', '9': 'nine', '10': 'ten'}
    new_seq = []
    for i in range(len(str_seq)):
        if(str_seq[i] in num_list):
            new_seq.append(num_list[str_seq[i]])
        else:
            new_seq.append(str_seq[i])
    return new_seq


def comb_WUPS(seq1, seq2):
    seq1_wup = []
    for a in seq1:
        temp_a = wn.synsets(a, pos=[wn.NOUN, wn.ADJ])
        if(len(temp_a)) != 0:
            seq1_wup.append(temp_a)

    seq2_wup = []
    for a in seq2:
        temp_a = wn.synsets(a, pos=[wn.NOUN, wn.ADJ])
        if (len(temp_a)) != 0:
            seq2_wup.append(temp_a)

    if (len(seq1_wup) != 0) and (len(seq2_wup) != 0):
        pair_score = np.zeros((len(seq1_wup), len(seq2_wup)))

        for i, interp_a in enumerate(seq1_wup):
            for j, interp_b in enumerate(seq2_wup):
                # for a pair
                global_max = 0.0
                for x in interp_a:
                    for y in interp_b:
                        local_score = x.wup_similarity(y)
                        if local_score > global_max:
                            global_max = local_score
                pair_score[i][j] = global_max

        score1 = np.prod(np.amax(pair_score, axis=0))
        score2 = np.prod(np.amax(pair_score, axis=1))
        WUPS_score = max([score1, score2])
        return WUPS_score

    else:
        WUPS_score = 0.0
        return WUPS_score


def build_similarity_list(imgs, N_base, list_1, list_2, similar_Q):
    WUPS_thres = 0.9
    print(similar_Q.shape)
    save_length = 10000

    for i, img in enumerate(imgs):
        local_pool_2 = []
        local_pool_2_original = []
        for j in range(save_length):
            if (list_2[i + N_base] != list_2[similar_Q[i + N_base][j]]) and (
                '-'.join(list_2[i + N_base][:-1]) not in '-'.join(list_2[similar_Q[i + N_base][j]][:-1])) and (
                '-'.join(list_2[similar_Q[i + N_base][j]][:-1]) not in '-'.join(list_2[i + N_base][:-1])) and (
                ''.join(list_2[i + N_base][:-1]) not in ''.join(list_2[similar_Q[i + N_base][j]][:-1])) and (
                ''.join(list_2[similar_Q[i + N_base][j]][:-1]) not in ''.join(list_2[i + N_base][:-1])) and (
                list_2[similar_Q[i + N_base][j]] not in local_pool_2):

                flag = 0
                for QQQ in range(len(local_pool_2)):
                    if ('-'.join(local_pool_2[QQQ][:-1]) not in '-'.join(list_2[similar_Q[i + N_base][j]][:-1])) and (
                        '-'.join(list_2[similar_Q[i + N_base][j]][:-1]) not in '-'.join(local_pool_2[QQQ][:-1])) and (
                        ''.join(local_pool_2[QQQ][:-1]) not in ''.join(list_2[similar_Q[i + N_base][j]][:-1])) and (
                        ''.join(list_2[similar_Q[i + N_base][j]][:-1]) not in ''.join(local_pool_2[QQQ][:-1])):
                        flag += 1
                    else:
                        break

                if flag == len(local_pool_2):
                    if comb_WUPS(list_2[i + N_base], list_2[similar_Q[i + N_base][j]]) <= WUPS_thres:
                        flag2 = 0
                        for QQQ in range(len(local_pool_2)):
                            if comb_WUPS(local_pool_2[QQQ], list_2[similar_Q[i + N_base][j]]) <= WUPS_thres:
                                flag2 += 1
                            else:
                                break

                        if flag2 == len(local_pool_2):
                            local_pool_2.append(list_2[similar_Q[i + N_base][j]])
                            local_pool_2_original.append(list_1[similar_Q[i + N_base][j]])

            if len(local_pool_2) == 3:
                break

        if i % 1000 == 0:
            print('in QA MC')
            print(i)

        img['Q_MC_token'] = []
        img['Q_MC_token_original'] = []






        if len(local_pool_2) < 3:
            print('poor img_Q_MC')
            print(i)
            for k in range(len(local_pool_2)):
                img['Q_MC_token'].append(local_pool_2[k])
                img['Q_MC_token_original'].append(local_pool_2_original[k])
            temp_order = np.random.permutation(3)
            for k in range(3 - len(local_pool_2)):

                img['Q_MC_token'].append(img['High_freq_A'][temp_order[k]])
                img['Q_MC_token_original'].append(img['High_freq_A_original'][temp_order[k]])


            img['Q_MC_token'].append(img['processed_Ans_tokens'])
            img['Q_MC_token_original'].append(img['Ans'])

        else:
            for k in range(3):
                img['Q_MC_token'].append(local_pool_2[k])
                img['Q_MC_token_original'].append(local_pool_2_original[k])
            img['Q_MC_token'].append(img['processed_Ans_tokens'])
            img['Q_MC_token_original'].append(img['Ans'])

    return imgs


def get_unqiue_img(imgs):
    WUPS_thres = 0.9
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    ques_pos_tmp = {}
    correct_ans = {}
    correct_ans_process = {}
    list_correct_ans = []
    list_correct_ans_process = []
    img_correct_ans = {}
    img_correct_ans_process = {}
    for img in imgs:
        count_img[img['Img_path']] = count_img.get(img['Img_path'], 0) + 1

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs):
        idx = imgtoi.get(img['Img_path'])
        img_pos[i] = idx

        if idx-1 not in ques_pos_tmp:
            ques_pos_tmp[idx-1] = []

        ques_pos_tmp[idx-1].append(i+1)
        correct_ans[i + 1] = img['Ans']
        correct_ans_process[i+1] = img['processed_Ans_tokens']
        list_correct_ans.append(img['Ans'])
        list_correct_ans_process.append(img['processed_Ans_tokens'])
    
    img_N = len(ques_pos_tmp)
    ques_pos = np.zeros((img_N, 300), dtype='uint32') # Harry: at most 50 questions of a image
    ques_pos_len = np.zeros(img_N, dtype='uint32')

    for idx, ques_list in ques_pos_tmp.iteritems():
        img_correct_ans[idx + 1] = []
        img_correct_ans_process[idx + 1] = []
        ques_pos_len[idx] = len(ques_list)
        for j in range(len(ques_list)):
            ques_pos[idx][j] = ques_list[j]
            if correct_ans_process[ques_list[j]] not in img_correct_ans_process[idx + 1]:
                img_correct_ans_process[idx + 1].append(correct_ans_process[ques_list[j]])
                img_correct_ans[idx + 1].append(correct_ans[ques_list[j]])

    for i, img in enumerate(imgs):
        local_pool = []
        local_pool_original = []
        temp_loc_order = np.random.permutation(len(img_correct_ans_process[img_pos[i]]))
        for j in range(len(img_correct_ans_process[img_pos[i]])):
            ok_ans = img_correct_ans_process[img_pos[i]][temp_loc_order[j]]
            if (correct_ans_process[i + 1] != ok_ans) and (
                '-'.join(correct_ans_process[i + 1][:-1]) not in '-'.join(ok_ans[:-1])) and (
                '-'.join(ok_ans[:-1]) not in '-'.join(correct_ans_process[i + 1][:-1])) and (
                ''.join(correct_ans_process[i + 1][:-1]) not in ''.join(ok_ans[:-1])) and (
                ''.join(ok_ans[:-1]) not in ''.join(correct_ans_process[i + 1][:-1])):

                flag = 0
                for QQQ in range(len(local_pool)):
                    if ('-'.join(local_pool[QQQ][:-1]) not in '-'.join(ok_ans[:-1])) and (
                        '-'.join(ok_ans[:-1]) not in '-'.join(local_pool[QQQ][:-1])) and (
                        ''.join(local_pool[QQQ][:-1]) not in ''.join(ok_ans[:-1])) and (
                        ''.join(ok_ans[:-1]) not in ''.join(local_pool[QQQ][:-1])):
                        flag += 1
                    else:
                        break

                if flag == len(local_pool):
                    if comb_WUPS(correct_ans_process[i + 1], ok_ans) <= WUPS_thres:
                        flag2 = 0
                        for QQQ in range(len(local_pool)):
                            if comb_WUPS(local_pool[QQQ], ok_ans) <= WUPS_thres:
                                flag2 += 1
                            else:
                                break

                        if flag2 == len(local_pool):
                            local_pool.append(ok_ans)
                            local_pool_original.append(img_correct_ans[img_pos[i]][temp_loc_order[j]])

            if len(local_pool) == 3:
                break

        if i % 1000 == 0:
            print('in IA MC')
            print(i)

        img['mod_MC_token'] = []
        img['mod_MC_token_original'] = []

        if len(local_pool) == 0:
            print('worst img mod_MC')
            print(img_pos[i])
            print(img['Ans'])


        if len(local_pool) < 3:
            print('poor img mod_MC')
            print(img_pos[i])
            for k in range(len(local_pool)):
                img['mod_MC_token'].append(local_pool[k])
                img['mod_MC_token_original'].append(local_pool_original[k])

            temp_order = np.random.permutation(3)
            for k in range(3 - len(local_pool)):

                img['mod_MC_token'].append(img['High_freq_A'][temp_order[k]])
                img['mod_MC_token_original'].append(img['High_freq_A_original'][temp_order[k]])


            img['mod_MC_token'].append(img['processed_Ans_tokens'])
            img['mod_MC_token_original'].append(img['Ans'])

        else:
            temp_order = np.random.permutation(len(local_pool))
            for k in range(3):
                img['mod_MC_token'].append(local_pool[temp_order[k]])
                img['mod_MC_token_original'].append(local_pool_original[temp_order[k]])
            img['mod_MC_token'].append(img['processed_Ans_tokens'])
            img['mod_MC_token_original'].append(img['Ans'])

    return unique_img, img_pos, ques_pos, ques_pos_len, imgs, list_correct_ans, list_correct_ans_process


def clean_img_func(imgs):
    clean_imgs = []
    for img in imgs:
        clean_img = {}
        if 'processed_tokens' in img:
            clean_img['processed_tokens'] = img['processed_tokens']

        if 'processed_Ans_tokens' in img:
            clean_img['processed_Ans_tokens'] = img['processed_Ans_tokens']

        if 'Q_MC_token' in img:
            clean_img['processed_MC_tokens'] = img['Q_MC_token']

        if 'mod_MC_token' in img:
            clean_img['IA_MC_tokens'] = img['mod_MC_token']

        if 'Q_MC_token' in img:
            clean_img['QA_MC_tokens'] = img['Q_MC_token']

        if 'High_freq_A' in img:
            clean_img['top_MC_tokens'] = img['High_freq_A']

        if 'Type' in img:
            clean_img['Type'] = img['Type']

        if 'ans_type' in img:
            clean_img['Type'] = img['ans_type']

        clean_imgs.append(clean_img)

    return clean_imgs


def clean_img_func_original(imgs):
    clean_imgs = []
    for img in imgs:
        clean_img = {}
        if 'Ques' in img:
            clean_img['question'] = img['Ques']

        if 'Ans' in img:
            clean_img['answer'] = img['Ans']

        if 'Q_MC_token_original' in img:
            clean_img['MC_answers'] = img['Q_MC_token_original']

        if 'mod_MC_token_original' in img:
            clean_img['IA_MC_answers'] = img['mod_MC_token_original']

        if 'Q_MC_token_original' in img:
            clean_img['QA_MC_answers'] = img['Q_MC_token_original']

        if 'High_freq_A_original' in img:
            clean_img['top_MC_answers'] = img['High_freq_A_original']

        if 'Type' in img:
            clean_img['Type'] = img['Type']

        if 'ans_type' in img:
            clean_img['Type'] = img['ans_type']

        clean_imgs.append(clean_img)

    return clean_imgs


def check_type(img):
    if img['processed_tokens'][0] == 'how':
        return 'how'
    elif img['processed_tokens'][0] == 'what':
        return 'what'
    elif img['processed_tokens'][0] == 'why':
        return 'why'
    elif img['processed_tokens'][0] == 'who':
        return 'who'
    elif img['processed_tokens'][0] == 'where':
        return 'where'
    elif img['processed_tokens'][0] == 'when':
        return 'when'
    else:
        return 'other'


def main(params):

    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_val = json.load(open(params['input_val_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    imgs_train = prepro_question(imgs_train, params)
    imgs_val = prepro_question(imgs_val, params)
    imgs_test = prepro_question(imgs_test, params)

    top_10_ans, top_10_ans_original = get_unqiue_A(imgs_train)
    print('high in train')
    imgs_train = add_frequent_A(imgs_train, top_10_ans, top_10_ans_original)
    print('high in val')
    imgs_val = add_frequent_A(imgs_val, top_10_ans, top_10_ans_original)
    print('high in test')
    imgs_test = add_frequent_A(imgs_test, top_10_ans, top_10_ans_original)

    out = {}
    out['train'] = imgs_train
    out['val'] = imgs_val
    out['test'] = imgs_test
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

    # get the unique image for train and test
    print('mod in train')
    unique_img_train, img_pos_train, ques_pos_train, ques_pos_len_train, imgs_train, list_1_train, list_2_train = get_unqiue_img(imgs_train)
    print('mod in val')
    unique_img_val, img_pos_val, ques_pos_val, ques_pos_len_val, imgs_val, list_1_val, list_2_val = get_unqiue_img(imgs_val)
    print('mod in test')
    unique_img_test, img_pos_test, ques_pos_test, ques_pos_len_test, imgs_test, list_1_test, list_2_test = get_unqiue_img(imgs_test)
    list_2 = list_2_train + list_2_val + list_2_test
    list_1 = list_1_train + list_1_val + list_1_test

    out = {}
    out['train'] = imgs_train
    out['val'] = imgs_val
    out['test'] = imgs_test
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

    file_h5h5 = h5py.File(params['input_h5'], 'r')
    similar_Q_total = file_h5h5['ques_similar_Q'][:]
    file_h5h5.close()
    print('Q_size')
    print(similar_Q_total.shape)
    print(len(list_1))

    imgs_train = build_similarity_list(imgs_train, 0, list_1, list_2, similar_Q_total)
    imgs_val = build_similarity_list(imgs_val, len(imgs_train), list_1, list_2, similar_Q_total)
    imgs_test = build_similarity_list(imgs_test, len(imgs_train) + len(imgs_val), list_1, list_2, similar_Q_total)

    for img in imgs_train:
        img['Type'] = check_type(img)
    for img in imgs_val:
        img['Type'] = check_type(img)
    for img in imgs_test:
        img['Type'] = check_type(img)

    # get the split
    N_train = len(imgs_train)
    N_val = len(imgs_val)
    N_test = len(imgs_test)
    split_train = np.zeros(N_train)
    split_val = np.zeros(N_val)
    split_val[:] = 1
    split_test = np.zeros(N_test)
    split_test[:] = 2

    split_train_v7w = np.zeros(N_train)
    split_val_v7w = np.zeros(N_val)
    split_test_v7w = np.zeros(N_test)

    count = 0
    for img in imgs_train:
        split_train_v7w[count] = img['In_v7w']
        count += 1
        if len(img['mod_MC_token']) != 4:
            print('mod_wrong')
        if len(img['Q_MC_token']) != 4:
            print('Q_wrong')
        if len(img['High_freq_A']) != 4:
            print('high_wrong')
    print(count)

    count = 0
    for img in imgs_val:
        split_val_v7w[count] = img['In_v7w']
        count += 1
        if len(img['mod_MC_token']) != 4:
            print('mod_wrong')
        if len(img['Q_MC_token']) != 4:
            print('Q_wrong')
        if len(img['High_freq_A']) != 4:
            print('high_wrong')
    print(count)

    count = 0
    for img in imgs_test:
        split_test_v7w[count] = img['In_v7w']
        count += 1
        if len(img['mod_MC_token']) != 4:
            print('mod_wrong')
        if len(img['Q_MC_token']) != 4:
            print('Q_wrong')
        if len(img['High_freq_A']) != 4:
            print('high_wrong')
    print(count)

    imgs_train_original = clean_img_func_original(imgs_train)
    imgs_val_original = clean_img_func_original(imgs_val)
    imgs_test_original = clean_img_func_original(imgs_test)

    imgs_train = clean_img_func(imgs_train)
    imgs_val = clean_img_func(imgs_val)
    imgs_test = clean_img_func(imgs_test)

    # create output h5 file for training set.
    f = h5py.File(params['output_h5'], "w")

    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    f.create_dataset("img_pos_val", dtype='uint32', data=img_pos_val)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)

    f.create_dataset("ques_pos_train", dtype='uint32', data=ques_pos_train)
    f.create_dataset("ques_pos_val", dtype='uint32', data=ques_pos_val)
    f.create_dataset("ques_pos_test", dtype='uint32', data=ques_pos_test)

    f.create_dataset("ques_pos_len_train", dtype='uint32', data=ques_pos_len_train)
    f.create_dataset("ques_pos_len_val", dtype='uint32', data=ques_pos_len_val)
    f.create_dataset("ques_pos_len_test", dtype='uint32', data=ques_pos_len_test)

    f.create_dataset("split_train", dtype='uint32', data=split_train)
    f.create_dataset("split_val", dtype='uint32', data=split_val)
    f.create_dataset("split_test", dtype='uint32', data=split_test)

    f.create_dataset("split_train_v7w", dtype='uint32', data=split_train_v7w)
    f.create_dataset("split_val_v7w", dtype='uint32', data=split_val_v7w)
    f.create_dataset("split_test_v7w", dtype='uint32', data=split_test_v7w)

    #f.create_dataset("ques_similar_Q", dtype='uint32', data=similar_Q_total)

    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['train'] = imgs_train_original
    out['val'] = imgs_val_original
    out['test'] = imgs_test_original
    json.dump(out, open(params['output_original'], 'w'))
    print 'wrote ', params['output_original']

    out = {}
    out['train'] = imgs_train
    out['val'] = imgs_val
    out['test'] = imgs_test
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='../data/VG_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_val_json', default='../data/VG_raw_val.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='../data/VG_raw_test.json', help='input json file to process into hdf5')

    parser.add_argument('--input_w2v_VG_h5', default='../data/w2v_VG.h5', help='input json file to process into hdf5')
    # parser.add_argument('--input_w2v_VG_Ans_h5', default='../data/w2v_VG_Ans.h5', help='input json file to process into hdf5')

    parser.add_argument('--input_h5', default='../data/VG_Q_1_similarity_1000.h5', help='input json file to process into hdf5')

    parser.add_argument('--output_original', default='../data/VG_data_prepro_augmented_oversets_A_WUPS_original.json', help='output json file')
    parser.add_argument('--output_json', default='../data/VG_data_prepro_augmented_oversets_A_WUPS_new.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/VG_data_prepro_augmented_oversets_A_WUPS_new.h5', help='output h5 file')
  
    # options
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')

    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
