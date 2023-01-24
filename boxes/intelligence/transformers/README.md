# Intelligence : Transformers

Using HuggingFace transformers and diffusers library to do some *insane* things...

## Requirements

1. Install HuggingFace libraries

```bash
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
pip install accelerate scipy
```

2. Run the following python code to "pre-download" (cache) the large models we will be using

```python
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Specify model
model_id = "stabilityai/stable-diffusion-2-1"

# Cache scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Cache diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
```

----
