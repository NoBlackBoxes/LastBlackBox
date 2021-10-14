Machine learning
================

Afternoon session

Goal
----

Create your own deep neural network (esp. convolutional neural network)
that can distinguish your face from other faces. Use the Coral USB
Accelerator for network inference and make the robot react to your face.

General note
------------

Typically, recognizing individuals is done differently, such that one
can identify multiple people. This fantastic work by Schroff et al.[^1]
is using something called “Triplet Loss”; in this session we are largely
simplifying this problem. However, the original idea is also in general
capable of re-identifying cells or neurons.

Task 1: Data collection
-----------------------

Get all your face data recorded and resized yesterday in a specific
folder, e.g. `my_faces`. We prepared a folder from the CelebA dataset[^2]
that contains in our case \~ 5,000 images of celebrities re-detected and
resized using the same algorithm as you are using for your face
detection. You can find this on [zenodo](https://zenodo.org/record/5561092#.YWQHg5pByUk).
Upload everything to [Google Colab](https://colab.research.google.com/).

Task 2: Train your own DNN
--------------------------

Use our `ENB train.ipynb` to train your own network. Go to [Google Colab](https://colab.research.google.com/) and upload it there and open the Jupyter notebook. It should be easy
going, however, try to understand what is happening: What are the
convolutional layers, what are these settings doing? Ensure you
understand:

-   Conv2D

-   Filters i.e. activation maps

-   Kernel size

-   Activation function (relu, sigmoid, …)

-   MaxPool2D

-   Flatten or GlobalAveragePooling2D

-   Dense layer

-   Epochs

Adjust Filters, Kernel Sizes, blocks of Conv2D and MaxPool2D operations
to see how the network changes (use `model.summary()`!) Train different
networks to see their learning behavior – specifically look at the loss
plots. Maybe train the network for longer… Try several things!

Task 3: Save and Convert and move your model to Edge TPU 
---------------------------------------------------------

Finally, save after model training your model using
`model.save(“your_model_name.h5”)`. Next, you need to deploy your model
to the Edge TPU. This is, however, quite complex. We prepared a script
(`ENB Keras to Edge TPU.ipynb`) for you that you can use on Google Colab
to convert your TensorFlow/Keras model to a TensorFlowLite/Edge TPU
version. Download the converted version, move it to your Raspberry Pi.

This is available through Colab:

<https://colab.research.google.com/drive/1mWneaXT_pd-MIO7jQn2s-UY8xHYE-8Qd?usp=sharing>

Task 4: Is it your face?
------------------------

Integrate your face recognition model into your live stream. If your
face is detected, maybe via the OpenCV face detector, it should be
colored green, if it is not your face (photograph, a friend, collegue,
PI, …) it should be colored differently. Let the robot react to your
face: Make the robot approach you when your face is detected. It should
escape, if not your face is detected. Estimate its specificity.

[^1]: <https://arxiv.org/abs/1503.03832>

[^2]: <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
