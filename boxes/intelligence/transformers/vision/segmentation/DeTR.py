import math
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForSegmentation

# Specify paths
repo = '/home/kampff/NoBlackBoxes/repos/LastBlackBox'
image_path = repo + '/boxes/intelligence/transformers/vision/_data/zoom_lesson.jpg'
image = Image.open(image_path)

# Display test image
plt.figure()
plt.imshow(image)
plt.show()

# Download feature extractor
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")

# Extract features (resizes and normalizes input image)
encoding = feature_extractor(image, return_tensors="pt")
print(encoding['pixel_values'].shape)

# Download model (172 MB)
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# Run model
outputs = model(**encoding)

# Compute the scores, excluding the "no-object" class (the last one)
scores = outputs.logits.softmax(-1)[..., :-1].max(-1)[0]

# Threshold the confidence
valid = scores > 0.45

# Plot all the valid masks
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(valid.sum().item() / ncols), figsize=(18, 10))
for line in axs:
    for a in line:
        a.axis('off')
for i, mask in enumerate(outputs.pred_masks[valid].detach().numpy()):
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask, cmap="cividis")
    ax.axis('off')
fig.tight_layout()
plt.show()

#FIN