# News-Recommender-Recsys24
RecSys @ University of Amsterdam '24

### Pre-Trained

Checkpoints: https://amsuni-my.sharepoint.com/:f:/g/personal/theofanis_aslanidis_student_uva_nl/EsgSoaTdU7xPinW1OS0MQTgB3OqYg5Lkqo_ih0rmTzgpuA?e=5fTr5b

Download and place in the folder ./checkpoints

### Tr4Rec Train



### Tr4Rec Evaluation

`python tr4rec/eval.py --split small --data_category validation --path ./checkpoints/enriched/small/checkpoint-226/pytorch_model.bin --dataset_type enriched`

### Extension Train & Test

`python tr4rec/extension_train_test.py --split small --extension_model attn --epochs 2`