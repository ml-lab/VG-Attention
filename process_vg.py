import os, sys
import os.path as osp
import json
from IPython import embed

qa_region_info_filepath = osp.join( 'data', 'qa_region_description.json' )

decoy_file_train    = osp.join('decoys', 'VG_train_decoys.json')
decoy_file_val      = osp.join('decoys', 'VG_val_decoys.json')
decoy_file_test     = osp.join('decoys', 'VG_test_decoys.json')

decoy_train_outfile = decoy_file_train.replace('.json', '.region_augmented.json')
decoy_val_outfile = decoy_file_val.replace('.json', '.region_augmented.json')
decoy_test_outfile = decoy_file_test.replace('.json', '.region_augmented.json')

with open(qa_region_info_filepath, 'r') as _in:
  qa2region = json.load(_in)

print '* processing train...'
augmented = list()
with open(decoy_file_train, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys)))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented.append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IoU_decoys'],
      'QoU_decoys': decoy['QoU_decoys'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['answer'],
      'question': decoy['question'],
      'region_id': region['region_id'],
      'phrase': region['phrase'],
      'height': region['height'],
      'width': region['width'],
      'x': region['x'],
      'y': region['y']
    })

  sys.stdout.write('\n')
  print '- writing in total {} decoys to {}'.format( len(augmented), decoy_train_outfile )
  with open(decoy_train_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)

print '* processing val...'
augmented = list()
with open(decoy_file_val, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys)))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented.append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IoU_decoys'],
      'QoU_decoys': decoy['QoU_decoys'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['answer'],
      'question': decoy['question'],
      'region_id': region['region_id'],
      'phrase': region['phrase'],
      'height': region['height'],
      'width': region['width'],
      'x': region['x'],
      'y': region['y']
    })

  sys.stdout.write('\n')
  print '- writing in total {} decoys to {}'.format( len(augmented), decoy_val_outfile )
  with open(decoy_val_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)


print '* processing test...'
augmented = list()
with open(decoy_file_test, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys)))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented.append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IoU_decoys'],
      'QoU_decoys': decoy['QoU_decoys'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['answer'],
      'question': decoy['question'],
      'region_id': region['region_id'],
      'phrase': region['phrase'],
      'height': region['height'],
      'width': region['width'],
      'x': region['x'],
      'y': region['y']
    })

  sys.stdout.write('\n')
  print '- writing in total {} decoys to {}'.format( len(augmented), decoy_test_outfile )
  with open(decoy_test_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)

