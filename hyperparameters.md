This file contains information about hyperparameters used to train our models

## Optimizer setup
- AdamW
    - betas (0.9, 0.98)
- Schedule
    - CosineAnnealingLR
        - T_max = no. of epochs
        - updates every epoch

## Finetuning models
Finetuning S example

```
python3 ft_selective.py --epochs=30 --batch_size=4 --accum=8 --pos_weight=3 --prot_info_path ../data/phosphosite_sequences/phosphosite_df.json --dataset_path=../data/splits_S.json --lr=5e-5 -o "$OUT_DIRNAME" --num_workers=15 --dropout=0.5 --indices="[-1, -2, -3]" --residues="['S']" --hidden_size=256
```
- analogically for T, Y and others
- for 3 layers, `--indices="[-1, -2, -3]"`, for 6 `--indices="[-1, -2, -3, -4, -5, -6]"`


## Encoder-based models
Encoder S example
```
python3 encoder.py --epochs=30 --batch_size=16 --accum=2 --pos_weight=3 --prot_info_path ../data/phosphosite_sequences/phosphosite_df.json --lr=3e-5 --num_workers=15 --dropout=0.5 --patience=15 --type="650M" --cnn_type='basic' --sr_n=3 --sr_final_size=256 --sr_kernel_size=3 --n_layers=1 --ffw_dim=2048 --n_layers_mlp=3 --dataset_path=../data/splits_S.json --res_kernel_size=3
```
- analogically for T, Y, and others
- `--residues` and `--dataset_path` according to the given model

## Encoder models with kinase info
Kinase S example
```
python3 kinase.py --epochs=30 --batch_size=16 --accum=2 --pos_weight=3 --prot_info_path ../data/phosphosite_sequences/phosphosite_df.json --lr=3e-5 -o "$OUT_DIRNAME" --num_workers=15 --dropout=0.5 --patience=15 --type="35M" --cnn_type='basic' --sr_n=3 --sr_final_size=256 --sr_kernel_size=3 --residues="['S']" --n_layers=1 --ffw_dim=2048 --n_layers_mlp=3 --kinase_emb_path="../data/kinase_embeddings_35M.pt" --kinase_info_path='../data/kinases_S.csv' --use_transform --nl_transform
```
- analogically for other residues