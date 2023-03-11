# stop_sign_detection
A project using OpenCV image preprocessing and Tensorflow + Keras machine learning to recognize stop signs.

## Image preprocessing

Input images are preprocessed extensively using the OpenCV library before being input into the model.

The model expects a 128x128 pixel image, with the sign in question (relatively) centered in the frame.
To achieve this, I treated a stop sign like a rough red circle.
First, a red color mask is extracted from the image to isolate instances of red. This is what the red mask looks like when extracted from ``input_img.jpg``:

![image](https://user-images.githubusercontent.com/58274830/224475634-28b34cd5-b1bc-46ad-99d2-202d61f9f8ad.png)

Then, the Hough circles algorithm is applied to grab any circle-like figures in the image.
Each circle is turned into a mask and applied to the red-extracted image, which leaves only the area of the image inside the circle as nonzero values.
For each one of these masked images, the nonzero values are counted and divided by circle area to obtain percent-nonzero area per circle.
Using the assumption that a stop sign is essentially a red circle, the most stop sign-like feature in the image would be something with high percent-nonzero area.
I assume only one stop sign per image, so the circle with the largest percent-nonzero area is chosen. This is what that looks like with ``input_img.jpg``:

![image](https://user-images.githubusercontent.com/58274830/224475911-c7ddb6ce-bbf2-4fba-b0db-210842f6b9d2.png)

Based on its position and radius, the image is cropped and resized for model input. On ``input_img.jpg``:

![image](https://user-images.githubusercontent.com/58274830/224475994-b9f8b12b-9678-4546-8600-0e057f5851e5.png)

Just this preprocessing could suffice as a stop sign detector if you were guaranteed that there are no other vaguely circular red objects that could ever be in frame.
However, I want to make sure that the thing I have identified is a stop sign, which is why I also have the model.

## The model
The model was trained using the Keras API and the Tensorflow backend.
Activations, mostly across the board, were implemented with Swish to prevent any disappearing gradients.
The last layer is activated with a Softmax classifier and optimized using sparse categorical cross entropy (mostly because the dataset I had was labeled with indexes instead of onehot vectors.

To account for training time and the simplicity of the task, the model architecture is relatively small (only around 1.2 million trainable weights):

```
Rescaling      (None, 128, 128, 3)       0
Conv2D         (None, 126, 126, 8)       224
Activation     (None, 126, 126, 8)       0
Conv2D         (None, 124, 124, 8)       584
Activation     (None, 124, 124, 8)       0
MaxPooling     (None, 62, 62, 8)         0
Conv2D         (None, 60, 60, 16)        1168
Activation     (None, 60, 60, 16)        0
Conv2D         (None, 58, 58, 16)        2320
Activation     (None, 58, 58, 16)        0
MaxPooling     (None, 29, 29, 16)        0
Conv2D         (None, 27, 27, 32)        4640
Activation     (None, 27, 27, 32)        0
Conv2D         (None, 25, 25, 32)        9248
Activation     (None, 25, 25, 32)        0
Flatten        (None, 4608)              0
Dense          (None, 256)               1179904
Dense          (None, 128)               32896
Dense          (None, 43)                5547
=================================================================
Total params: 1,236,531
Trainable params: 1,236,531
Non-trainable params: 0
```

The model was trained upon the GTSRB traffic sign dataset (around 39,232 training images) in batches of 32 images for 10 epochs, and achieved ~95% accuracy on the test set.
The accuracy could definitely be improved with more training time, but there was no point for this project.

## Final product
The final pipeline is able to correctly determine the existence of a stop sign in both ``input_img.jpg`` and ``input_img_2.jpg`` (signified by printing ``14``, the class of stop signs in the GTSRB dataset)!

It is also able to correctly determine that there is no stop sign present in ``input_img_3.jpg``, despite the presence of a large red stop light.
