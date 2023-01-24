# Learning : transfer

Using PyTorch to retrain a CNN "backbone" network

## Requirements

1. Install pytorch and tools

```bash
pip install torch torchvision torchsummary
```

2. Donwload COCO dataset: 2017 training images and annotations and extract to a "coco" folder.

  - [2017 Training Images](http://images.cocodataset.org/zips/train2017.zip)
  - [2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

3. Install python COCO tools

```bash
pip install pycocotools
```

## Backbones

1. Use PyTorch to run MobileNet (version 2): [backbone](backbone/backbone_mobilenet.py)

2. Try different backbone models, different images...

    - *challenge*: Run a  "live" classifier by streaming images from your camera using OpenCV

## Tracking

1. Train a nose tracker by editing a backbone CNN and retraining it on "nose" locations (i.e. keypoints) in the COCO dataset.

- The PyTorch models are stored in a model.py file, the dataset in a datset.py file, and training is run by the trin.py script. *NOTE*: Thesescripts assume you running python from the folder containing all of these files (i.e. correct working directory).

  - [regression training](tracking/regression/train.py)
  - [heatmap training](tracking/heatmap/train.py)

----
