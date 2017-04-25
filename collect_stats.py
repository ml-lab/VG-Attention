import os, sys
import os.path as osp
import json
from IPython import embed

qa_region_info_filepath = osp.join( 'data', 'qa_region_description.json' )

decoy_file_train    = osp.join('decoys', 'VG_train_decoys.json')
decoy_file_val      = osp.join('decoys', 'VG_val_decoys.json')
decoy_file_test     = osp.join('decoys', 'VG_test_decoys.json')

with open(qa_region_info_filepath, 'r') as _in:
  qa2region = json.load(_in)

print '* processing train...'
tot_width, tot_height, tot_num = 0, 0, 0
augmented = list()
with open(decoy_file_train, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys)))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]

    tot_height = tot_height + region['height']
    tot_width  = tot_width + region['width']
    tot_num    = tot_num + 1

  sys.stdout.write('\n')

print '* processing val...'
augmented = list()
with open(decoy_file_val, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys)))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]

    tot_height = tot_height + region['height']
    tot_width  = tot_width + region['width']
    tot_num    = tot_num + 1

  sys.stdout.write('\n')


print '* processing test...'
augmented = list()
with open(decoy_file_test, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys)))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]

    tot_height = tot_height + region['height']
    tot_width  = tot_width + region['width']
    tot_num    = tot_num + 1

  sys.stdout.write('\n')

print 'average_height: {}, average_width: {}'.format(tot_height / float(tot_num), tot_width / float(tot_num))
