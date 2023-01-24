import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image

# Specify paths
coco_folder = '/home/kampff/Dropbox/Voight-Kampff/Technology/Datasets/coco'
dataset_name = 'val2017'
annotations_path = coco_folder + '/annotations/person_keypoints_val2017.json'
images_path = coco_folder + '/val2017'

# Initialize the COCO API
coco=COCO(annotations_path)

# Select all people: category and image IDs
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids )

# Keypolint labels
keypoint_lables = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']

# Select annotations of images with only one person with a visible nose
valid_annotations = []
for img in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img, catIds=cat_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    # Individuals
    if len(annotations) > 1:
        continue

    # Not too small
    if (annotations[0]['area'] < 10000):
        continue

    # No crowds
    if annotations[0]['iscrowd']:
        continue

    # Only visible noses
    if (annotations[0]['keypoints'][2] == 0):
        continue
    
    # Include
    valid_annotations.append(annotations[0])

# Display random examples
for i in range(9):
    plt.subplot(3,3,i+1)
    index = np.random.randint(0,len(valid_annotations))
    image_id = valid_annotations[index]['image_id']
    img = coco.loadImgs(image_id)[0]
    image = Image.open(images_path + '/' + img['file_name'])
    plt.imshow(image); plt.axis('off')
    coco.showAnns([valid_annotations[index]])
    nose_X = valid_annotations[index]['keypoints'][0]
    nose_Y = valid_annotations[index]['keypoints'][1]
    plt.plot(nose_X, nose_Y, 'r+')
plt.show()

#FIN