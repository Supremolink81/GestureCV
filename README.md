# GestureCV (Work In Progress)
Deep Learning + Computer Vision project to classify one of 4 hand gestures in images/video.

Uses a CNN for Image Classification.

(This was originally an object detection project, but due to time and development constraints I changed it to a purely classification based project).

# Lessons Learned From This

Just as a preliminary to this, here are some very important lessons I learned from this project:

- do not gather your own data unless the data you need doesn't already exist and you have expertise in data collection practices

- training a neural network from scratch is not a good idea; look for pretrained options first.

- try to find simple architectures first, then work your way up if your model is underfitting when being trained

- do not use unbalanced datasets!

# Data Collection

A total of 1500 image samples were collected, according to the following format:

- 300 photos taken over 5 locations.
- In each location, 50 samples were allocated for each gesture, with 100 being control samples (no gesture).
- For the control samples, 25 had a face, 25 had a hand, 25 had neither, and 25 had both.
- For the gestures, 25 were recorded for each hand in various positions, with approximately half (12-13) of the 25 including faces, with the other half not including any face.
- the gestures recorded are: the middle finger, the ok sign, the thumbs up, and the peace sign (two fingers up)

A few sample images are shown below:

![First Sample Image](./IMG_16640.png)
![Second Sample Image](./IMG_02570.png)
![Third Sample Image](./IMG_10920.png)

Photos were taken using an iPhone 13 Pro in the HEIC format, with 2316x3088 pixel resolution. When ported to a Windows PC, their resolution changed, with some images being 756x1008 and some being 579x772.

Through a flip along the x axis for the 4 gesture classes, the dataset was expanded to 2500 images, 500 for each class.

# Issues around the data collection

The largest issue is that a single person (myself) was used to take the photos; the model has learned to recognize gestures from people with my hand size, structure, and skin color. For example, if someone with darker skin and a larger hand than mine performed gestures, the model may struggle to recognize their gestures. I predict a more complex architecture than the one ultimately chosen would need to be used to effectivly capture these differences.

Another issue is the size of the dataset; there are only 1500 samples collected, and even after preprocessing (described in the next section), 12000 samples are available. This is far less than datasets used to train state-of-the-art models (such as ImageNet), which contain millions of samples.

Finally, the size of the images; the photos are far larger than popular image dataset sizes (MNIST, for instance, uses 28x28 photos [1]).

# Preprocessing

Preprocessing the data involves a few steps: we first convert the HEIC images to PNG. To ensure all images are the same size, we resize them to a size of 126x126. Then, to increase the size of our dataset, we will create more training examples by taking transformations of the images. 

In this case, we take 90, 180 and 270 degree rotations, then apply Gaussian noise to the images. This results in 8 images generated from 1 original sample, for a total of 12000 images generated from the original data.

We then split the data into 3 groups; 60% being the training data, 20% being the validation data, and 20% being the test data. In each subset, 1/3 of the data contains no gesture, while 1/6 of the subset is allocated for each gesture.

Sample transformed images:

![First Sample Transformed Image](./img10.png)
![Second Sample Transformed Image](./img23.png)
![Third Sample Transformed Image](./img51.png)

# Choosing An Architecture For The Classifier

To start, the ResNet18 model architecture is chosen; ResNet performs well in image classification [2], and is readily available
through torchvision (disclaimer: torchvision's ResNet implementations do not match the original paper's; torchvision's has significantly more parameters).

If needed, we have access to the ResNet34 and ResNet50 architectures, among others, through torchvision.

# Choosing an optimizer and loss function

The loss function we will use is softmax cross entropy loss; this is a common loss used for image classification [3].

The optimizer we will use is stochastic gradient descent, as this was used by the original ResNet with great success [2].

# Hyperparameter Tuning For The Classifier

For more details regarding this, see the Trials.md file.

(In progress...)

# References

[1] https://en.wikipedia.org/wiki/MNIST_database

[2] https://arxiv.org/abs/1512.03385

[3] https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

[4] https://cs231n.github.io/neural-networks-3/#loss

[5] https://github.com/kuangliu/pytorch-cifar/issues/136

[6] https://arxiv.org/abs/1807.11164

[7] https://arxiv.org/pdf/1807.11164.pdf

[8] https://arxiv.org/abs/1707.01083

[9] http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

[10] https://stats.stackexchange.com/questions/303857/explanation-of-spikes-in-training-loss-vs-iterations-with-adam-optimizer/304150#304150

[11] https://www.researchgate.net/figure/ResNet-9-architecture-A-convolutional-neural-net-with-9-layers-and-skip-connections_fig1_363585139