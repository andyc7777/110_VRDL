# 2021_VRDL_HW3

## Setup
```
git clone https://github.com/cocodataset/cocoapi 

cd cocoapi/PythonAPI

python3 setup.py build_ext install  
make
```

```
cd ../../
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
pip3 install -r requirements.txt
python3 setup.py install
```
```
wget -c https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 \
      -O Mask_RCNN/mask_rcnn_coco.h5
```
```
pip3 uninstall keras -y
pip3 uninstall keras-nightly -y
pip3 uninstall keras-Preprocessing -y
pip3 uninstall keras-vis -y
pip3 uninstall tensorflow -y
pip3 uninstall tensorflow-gpu -y
pip3 uninstall h5py -y
pip3 uninstall scikit-image

pip3 install keras==2.0.8
pip3 install h5py==2.10.0
pip3 install -U scikit-image==0.16.2
pip3 install tensorflow==1.15.2
pip3 install tensorflow-gpu==1.15.2
```
```
cd Mask_RCNN/samples
!rm -rf 2021_VRDL_HW3
!git clone https://github.com/andychan8877/2021_VRDL_HW3.git

!cp -r /content/cocoapi/PythonAPI/pycocotools /content/Mask_RCNN/samples/2021_VRDL_HW3/
```

## Command Line
### To train model using imagenet pretrained weight
```
cd Mask_RCNN/samples/2021_VRDL_HW3
python3 hw3_nuclei.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```
### Inference
```
cd Mask_RCNN/samples/2021_VRDL_HW3
python3 inference.py inference --dataset=/path/to/dataset --subset=test --weights=/path/to/weight.h5
```

or

```
cd Mask_RCNN/samples/2021_VRDL_HW3
python3 hw3_nuclei.py detect --dataset=/path/to/dataset --subset=test --weights=/path/to/weight.h5
```
