import os, sys
import os.path as osp
import json
from IPython import embed

qa_region_info_filepath = osp.join( 'data', 'qa_region_description.json' )

decoy_file = osp.join('decoys', 'VG_decoys_tokenized.json')

decoy_outfile = decoy_file.replace('.json', '.{}.region_augmented.json')

with open(qa_region_info_filepath, 'r') as _in:
  qa2region = json.load(_in)

print '* processing train...'
augmented = list()
with open(decoy_file, 'r') as decoy_:
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys['train']):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys['train'])))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented.append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IA_MC_tokens'],
      'QoU_decoys': decoy['QA_MC_tokens'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['processed_Ans_tokens'],
      'question': decoy['processed_tokens'],
      'region_id': region['region_id'],
      'phrase': region['phrase'],
      'height': region['height'],
      'width': region['width'],
      'x': region['x'],
      'y': region['y']
    })

  decoy_train_outfile = decoy_outfile.format('train')
  sys.stdout.write('\n')
  print '- writing in total {} decoys to {}'.format( len(augmented), decoy_train_outfile )
  with open(decoy_train_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)

  print '* processing val...'
  augmented = list()
  for _idx, decoy in enumerate(decoys['val']):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys['val'])))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented.append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IA_MC_tokens'],
      'QoU_decoys': decoy['QA_MC_tokens'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['processed_Ans_tokens'],
      'question': decoy['processed_tokens'],
      'region_id': region['region_id'],
      'phrase': region['phrase'],
      'height': region['height'],
      'width': region['width'],
      'x': region['x'],
      'y': region['y']
    })

  sys.stdout.write('\n')
  decoy_val_outfile = decoy_outfile.format('val')
  print '- writing in total {} decoys to {}'.format( len(augmented), decoy_val_outfile )
  with open(decoy_val_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)


  print '* processing test...'
  augmented = list()
  for _idx, decoy in enumerate(decoys['test']):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys['test'])))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented.append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IA_MC_tokens'],
      'QoU_decoys': decoy['QA_MC_tokens'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['processed_Ans_tokens'],
      'question': decoy['processed_tokens'],
      'region_id': region['region_id'],
      'phrase': region['phrase'],
      'height': region['height'],
      'width': region['width'],
      'x': region['x'],
      'y': region['y']
    })

  sys.stdout.write('\n')
  decoy_test_outfile = decoy_outfile.format('test')
  print '- writing in total {} decoys to {}'.format( len(augmented), decoy_test_outfile )
  with open(decoy_test_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)

