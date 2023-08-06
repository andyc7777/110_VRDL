# 2021_VRDL_HW4

## Setup
```
git clone https://github.com/cszn/KAIR.git
pip install timm
```
#### Put vrdl_hw4_train.py in KAIR
#### Put inference.py in KAIR
#### Put vrdl_hw4_train.json in KAIR/options/swinir

## Training
```
cd KAIR

python vrdl_hw4_train.py --opt options/swinir/vrdl_hw4_train.json

```
### Inference Model Weight
https://drive.google.com/file/d/158Lj8p-jYrZsJnTZMW1951QTMLJRc0EP/view?usp=sharing

## Inference
```
cd KAIR

python inference.py --task classical_sr --scale 3 --training_patch_size 48 --model_path model_path --folder_lq lr_images_path --folder_gt lr_images_path

```
