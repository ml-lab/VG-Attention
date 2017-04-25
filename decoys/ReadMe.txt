(1) We provide our generated decoys for Visual Genome (v1.2) in 3 json files:

VG_train_decoys.json 
VG_val_decoys.json 
VG_test_decoys.json

We split the dataset into 50%/20%/30% for training/validation/test. We partition such that each portion is a “superset” of the corresponding one in Visual7W, respectively.



(2) Each json file contains a list. Each item of the list is like follows,

{"QoU_decoys": ["Blue.", "Black.", "White."], "Img_id": 2, "question": "What color is the car?", "QA_id": 986934, "answer": "Red.", "IoU_decoys": ["A backpack.", "Glass.", "Parked on the street."], "Type": "what"}.

"QoU_decoys": a list of 3 QoU decoys.
"IoU_decoys": a list of 3 IoU decoys.
"Img_id": follow the original dataset.
"QA_id": follow the original dataset.
"Type": question type (how/where/what/why/when/who/other).
"question": the question in the original dataset.
"answer": the correct answer (target) in the original dataset.

The "question", "answer", "Img_id", and "QA_id" follow the original dataset.
The original dataset can be downloaded from: 
Visual Genome (v1.2): http://visualgenome.org/api/v0/api_home.html



(3) If you use our datasets for any published research, it would be nice if you would cite our paper as well as the corresponding paper of original dataset.

@inproceedings{krishnavisualgenome,
  title={Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations},
  author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and Bernstein, Michael and Fei-Fei, Li},
  year = {2016},
  url = {https://arxiv.org/abs/1602.07332},
}