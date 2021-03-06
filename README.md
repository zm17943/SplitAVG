# SplitAVG
This repository includes the official project of SplitAVG, from our paper "SplitAVG: A heterogeneity-aware federated deep learning method for medical imaging"(https://arxiv.org/pdf/2107.02375.pdf), accepted to IEEE Journal of Biomedical and Health Informatics (JBHI) 2022.


## Data preparation

Our simulated federated data partitions for bone age regression task can be downloaded at: https://drive.google.com/drive/folders/1O8oJ2KBHUprvRo0ILD93XfiYaW8u0Q9I?usp=sharing

The training data consists of 4 hetergeneous splits (clients). The organization structures are:

```
boneS1.h5
  -- ['examples']  -- ['0']   -- ['pixels']
                              -- ['label']
                   -- ['1']   -- ['pixels']
                              -- ['label']    
                   -- ['2']   -- ['pixels']
                              -- ['label'] 
                   -- ['3']   -- ['pixels']
                              -- ['label']

val.h5
  -- ['examples']  -- ['0']   -- ['pixels']
                              -- ['label']
```

Visualization of the heterogeneity in label distributions:

<img width="360" alt="Screen Shot 2022-07-12 at 2 25 47 PM" src="https://user-images.githubusercontent.com/30038903/178567298-c4f196c3-86b5-4312-a630-cb475c1a4816.png">


## Train
```
python main_SplitAVG.py \
--batch_size 32 \
--train_file "./data/boneS1.h5" --val_file "./data/val.h5" --site_num 4 \
--arch 'res34' \
--splitavg_v2 False \
--seed 2556 \
```
