import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from pycocotools.coco import COCO

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None, target_transform=None, augment=False):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        target = self.targets[idx,:]

        # Adjust image color
        if image.shape[2] == 1: # Is grayscale?
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augment (or just resize)
        if self.augment:
            image, target = augment(image, target)
        else:
            image = cv2.resize(image, (224,224))

        # Generate heatmap
        heatmap = generate_heatmap(target)

        # Set transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, heatmap

# Load dataset
def prepare(dataset_name, split):

    # Filter train and test datasets
    image_paths, targets = filter(dataset_name)

    # Split train/test
    num_samples = len(targets)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = (image_paths[train_indices], targets[train_indices])
    test_data = (image_paths[test_indices], targets[test_indices])

    return train_data, test_data

# Filter dataset
def filter(dataset_name):

    # Specify paths
    coco_folder = '/home/kampff/Dropbox/Voight-Kampff/Technology/Datasets/coco'
    annotations_path = coco_folder + '/annotations/person_keypoints_' + dataset_name + '.json'
    images_folder = coco_folder + '/' + dataset_name

    # Initialize the COCO API
    coco=COCO(annotations_path)

    # Select all people: category and image IDs
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids )

    # Select annotations of images with only one person with a visible nose
    image_paths = []
    targets = []
    for i in img_ids:
        ann_ids = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)

        # Individuals
        if len(annotations) > 1:
            continue

        # No crowds
        if annotations[0]['iscrowd']:
            continue

        # Extract relevant keypoints
        keypoints =     annotations[0]['keypoints']
        nose_x =        keypoints[0]
        nose_y =        keypoints[1]
        nose_visible =  keypoints[2]
        l_eye_x =       keypoints[3]
        l_eye_y =       keypoints[4]
        l_eye_visible = keypoints[5]
        r_eye_x =       keypoints[6]
        r_eye_y =       keypoints[7]
        r_eye_visible = keypoints[8]

        # Visible nose and eyes
        if (nose_visible == 0) or (l_eye_visible == 0) or (r_eye_visible == 0):
            continue
        
        # Big face
        eye_distance = abs(l_eye_x - r_eye_x) + abs(l_eye_y - r_eye_y)
        if eye_distance < 40:
            continue

        # Isolate image path
        img = coco.loadImgs(annotations[0]['image_id'])[0]

        # Normalize nose centroid
        width = img['width']
        height = img['height']
        x = np.float32(nose_x) / width
        y = np.float32(nose_y) / height
        target = np.array([x, y], dtype=np.float32)

        # Store dataset
        image_paths.append(images_folder + '/' + img['file_name'])
        targets.append(target)

    return np.array(image_paths), np.array(targets)

# Augment
def augment(image, target):

    # Image size
    height, width, depth = image.shape

    # Nose position (pixels)
    nose_x = width * target[0]
    nose_y = height * target[1]

    # Measure buffering
    cushion = 30
    buffer_left = int(nose_x) - cushion
    buffer_right = width - int(nose_x) - cushion
    buffer_top = int(nose_y) - cushion
    buffer_bottom = height - int(nose_y) - cushion

    # Clamp
    buffer_left = np.clip(buffer_left, 1, width)
    buffer_right = np.clip(buffer_right, 1, width)
    buffer_top = np.clip(buffer_top, 1, height)
    buffer_bottom = np.clip(buffer_bottom, 1, height)

    # Set crop window
    crop_left = np.random.randint(0, buffer_left)
    crop_right = width - np.random.randint(0, buffer_right)
    crop_top = np.random.randint(0, buffer_top)
    crop_bottom = height - np.random.randint(0, buffer_bottom)
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    # Crop
    crop = image[crop_top:crop_bottom, crop_left:crop_right, :]

    # Augment (resize cropped, offset and scale target)
    augmented_image = cv2.resize(crop, (224,224))
    augmented_target = np.zeros(2, dtype=np.float32)
    augmented_target[0] = (nose_x - crop_left) / crop_width 
    augmented_target[1] = (nose_y - crop_top) / crop_height

    return augmented_image, augmented_target

# Generate heatmap
def generate_heatmap(target):
    heatmap = np.zeros((224,224), dtype=np.float32)
    x = target[0]
    y = target[1]
    ix = int(np.floor(x * 224))
    iy = int(np.floor(y * 224)) 
    heatmap[iy][ix] = 1.0
    heatmap = cv2.GaussianBlur(heatmap, ksize=(51,51), sigmaX=9, sigmaY=9)
    heatmap = cv2.resize(heatmap, (14,14), interpolation=cv2.INTER_LINEAR)
    heatmap = heatmap / np.sum(heatmap[:])
    heatmap = np.expand_dims(heatmap, axis=0)

    return heatmap

#FIN