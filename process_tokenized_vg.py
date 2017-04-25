import os, sys
import os.path as osp
import json
from IPython import embed

qa_region_info_filepath = osp.join( 'data', 'qa_region_description.json' )

decoy_file = osp.join('decoys', 'VG_decoys_tokenized.json')

decoy_outfile = decoy_file.replace('.json', '.region_augmented.json')

with open(qa_region_info_filepath, 'r') as _in:
  qa2region = json.load(_in)

print '* processing train...'
augmented = dict()
with open(decoy_file, 'r') as decoy_:
  augmented['train'] = list()
  decoys = json.load(decoy_)
  for _idx, decoy in enumerate(decoys['train']):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys['train'])))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented['train'].append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IA_MC_tokens'],
      'QoU_decoys': decoy['QA_MC_tokens'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['processed_Ans_tokens'],
      'question': decoy['processed_tokens']
    })

  decoy_train_outfile = decoy_outfile.format('train')
  sys.stdout.write('\n')

  print '* processing val...'
  augmented['val'] = list()
  for _idx, decoy in enumerate(decoys['val']):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys['val'])))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented['val'].append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IA_MC_tokens'],
      'QoU_decoys': decoy['QA_MC_tokens'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['processed_Ans_tokens'],
      'question': decoy['processed_tokens']
    })

  sys.stdout.write('\n')

  print '* processing test...'
  augmented['test'] = list()
  for _idx, decoy in enumerate(decoys['test']):
    sys.stdout.write('- processing image {}/{}\r'.format(_idx, len(decoys['test'])))
    sys.stdout.flush()
    if qa2region.get( str(decoy['QA_id']), None ) == None: continue
    region = qa2region[ str(decoy['QA_id']) ]
    augmented['test'].append({
      'image_id': decoy['Img_id'],
      'IoU_decoys': decoy['IA_MC_tokens'],
      'QoU_decoys': decoy['QA_MC_tokens'],
      'QA_id': decoy['QA_id'],
      'type': decoy['Type'],
      'answer': decoy['processed_Ans_tokens'],
      'question': decoy['processed_tokens']
    })

  sys.stdout.write('\n')
  print '- writing in total decoys to {}'.format(decoy_outfile )
  with open(decoy_outfile, 'w') as decoy_:
    json.dump(augmented, decoy_)

