# Trials

Our goal is to gain 90% accuracy or greater on the validation set.

To track both underfitting and overfitting, we will display both the training and validation accuracy.

As a start, we will tune only the learning rate, batch size and epochs, leaving the rest of SGD's hyperparameters at 0; if greater accuracy is desired, we can introduce SGD's other hyperparameters after tuning the former 3.

To start with, we will visualize the loss value over batch iterations, then, if/when we need to perform extremely fine-tuned optimization, we can use a parameter search technique, such as random or grid search.

A visualization of ideal vs non-ideal learning rates [4]:

![Ideal Learning Rate Graph](./ideal_learning_rate_graph.png)

We start with a learning rate of 0.001, a batch size of 50, and 10 epochs; this yields 35.986% training accuracy and 33.917% validation accuracy. The graph of the loss is below:

![Loss Function Graph 1](./loss_function_graph_1.png)

We gain 3 insights from this; first off, comparing to the learning rate graph from [4], we see that the learning rate is too high. In addition, due to the high fluctuation of the loss, we can conclude the batch size is too low [4]. Finally, from the accuracies, the model is heavily underfitting; we can solve this by increasing the epochs. 

The new hyperparameters we will use are a learning rate of 0.0001, a batch size of 100, and 15 epochs. This yields 33.431% training accuracy and 33.208% validation accuracy, with the following loss graph:

![Loss Function Graph 2](./loss_function_graph_2.png)

Noticeable improvement in the learning rate curve, but still too drastic. The fluctuation is better initially, but degrades to the same level of chaos after the learning rate plateaus. A larger batch size is still needed. The model is also still underfitting. 

Now we try a learning rate of 0.00001, a batch size of 150, and 50 epochs. This yields 33.042% training accuracy and 32.833% validation accuracy, with the following loss graph:

![Loss Function Graph 3](./loss_function_graph_3.png)

Hm...that didn't seem to work either. Perhaps the learning rate is too low? Let's try a learning rate of 0.001 once again, with 50 epochs and a batch size of 150. This yields a 60.806% training and 38.417% validation accuracy, with the following loss graph:

![Loss Function Graph 4](./loss_function_graph_4.png)

Now we're getting somewhere! Since it seemed like increasing the learning rate was effective, let's try a learning rate of 0.005, 50 epochs and a batch size of 200 to decrease oscillation. This yields 100.0% training accuracy and 67.792% validation accuracy, with the following loss graph:

![Loss Function Graph 5](./loss_function_graph_5.png)

Much better, but it seems the learning rate is still a bit too low given what we see from the curve (the good news, though, is there is vry little oscillation, our optimal batch size seems to be 200). In addition, it seems that the model is overfitting to the training data. generally, there are two possible reasons for this:

- the training data was unrepresentative

- the model architecture was too complex

Due to all 3 sets having the same distribution of classes, the first one is unlikely; however, the second one is plausible, as Resnet18, despite being one of the smaller ResNets, has over 11 million parameters [5]. There exists an alternative architecture, ShuffleNetV2[6], with approximately 346917 under PyTorch's implementation, so it is much less prone to overfitting in terms of model complexity. In addition, it trains about 13 times faster than AlexNet and achieves comparable accuracy [8]. 

We set the learning rate to 0.01, batch size to 200, and epochs to 50, and our new architecture of ShuffleNetV2 (we use the 1.0x version described in the paper). This yields 50.05555555555556% training accuracy and 39.083333333333336% validation accuracy, with the following loss graph:

![Loss Function Graph 6](./loss_function_graph_6.png)

Alright, seems we need to increase the learning rate significantly, as well as increase batch size to reduce oscillation. In addition, we will increase the number of epochs, as the model may still be underfitting. We now train with a learning rate of 0.1, a batch size of 300, and 75 epochs. This yields 100.0% training accuracy and 72.25% validation accuracy, with the following loss graph:

![Loss Function Graph 7](./loss_function_graph_7.png)

Now we're really getting somewhere! Our validation accuracy is the best yet, and it seems there were many unnecessary epochs (though, batch size and learning rate appear to be fine). We reduce epochs to 30, keeping learning rate at 0.1 and batch size at 300, resulting in 99.556% training accuracy and 69.083% validation, with the following loss graph:

![Loss Function Graph 8](./loss_function_graph_8.png)

It appears this architecture suffers from the same issue as ResNet18; too complex, thus it is overfitting. After some searching, we find the LeNet-5 architecture [9], a relatively small CNN and one of the first major architectures used for MNIST. This architecture has 60 000 parameters (less than 3% of ShuffleNet's 2 million), meaning it will probably overfit much less. There are some modifications we have made; we grayscale the images due to LeNet-5 only accepting 1 color channel. As well, due to the difference in image sizes (126x126 vs the 32x32 LeNet-5 originally expects), the first fully connected layer has 12544 inputs rather than 256.

As a safe default, we will set the learning rate to 0.001, the batch size to 50, and the epochs to 50. This yields 33.333% training accuracy and 33.333%, with the following loss graph:

![Loss Function Graph 9](./loss_function_graph_9.png)

Learning rate is definitely too low, and the batch size is far too low. We now try a 0.01 learning rate, 150 batch size, and 50 epochs. This yields , with the following loss graph:

![Loss Function Graph 10](./loss_function_graph_10.png)

Hm...that didn't seem to do much. Since the batch size is relatively large, it is probably the learning rate being too high. At this point, to speed up trials, we will do a grid search over learning rate values. The values we will pick are 0.001, 0.0005, 0.0001, 0.00005 and 0.00001. To be safe, we set the batch size to 200, and we keep the epochs at 50. This yields the following loss graphs and accuracies:

Learning rate = 0.001: 33.333% training accuracy and 33.333% validation accuracy.

![Loss Function Graph 11](./loss_function_graph_11.png)

Better...

Learning rate = 0.0005: 33.333% training accuracy and 33.333% validation accuracy.

![Loss Function Graph 12](./loss_function_graph_12.png)

Not better.

Learning rate = 0.0001: 16.708% training accuracy and 16.167% validation accuracy.

![Loss Function Graph 13](./loss_function_graph_13.png)

Learning rate = 0.00005: 16.667% training accuracy and 16.667% validation accuracy.

![Loss Function Graph 14](./loss_function_graph_14.png)

Learning rate = 0.00001: 16.667% training accuracy and 16.667% validation accuracy.

![Loss Function Graph 15](./loss_function_graph_15.png)

Evidently, the model's training is the most stable at a learning rate of 0.001. Though, it seems the model has not had nearly enough epochs to train properly. In addition, there is still lots of variance in the gradient updates, and since learning rate is not the issue (we know this because variance only increased as we increased learning rate), this suggests that the batch size is too small. Thus, we will try a learning rate of 0.001, a batch size of 400, and 500 epochs of training to see what happens. This yields 33.333% training accuracy and 33.333%, with the following loss graph (note, after this point, I decided to grayscale the images to improve not only memory constraints, but time constraints as well. This means the modification from earlier where the input channels was increased to 3 was changed back to 1.):

![Loss Function Graph 16](./loss_function_graph_16.png)

Ok, somewhat promising. Our loss decreases for some time, then starts to oscillate. Perhaps the learning rate is too high? Trying a learning rate of 0.00001, a batch size of 400, and 500 epochs, yields , with the following loss graph:

![Loss Function Graph 17](./loss_function_graph_17.png)

Okay, something weird is going on. Time for Googling.

Yup, found the problem; exploding gradients. Turns out, at some points of optimization, the gradient gets so large an oscillating effect happens. The solution to this is to induce a cap on the gradient norm. In our case, we will cap the norm at 2.

We now once again try a learning rate of 0.001, a batch size of 400, and 500 epochs. This yields 33.333% training accuracy and 33.333% validation accuracy, with the following loss graph:

![Loss Function Graph 18](./loss_function_graph_18.png)

Perhaps that didn't clip the gradients properly. Let's try another method:

```py
for parameter in self.model.parameters():

    parameter.register_hook(lambda gradient: torch.clamp(gradient, -1.0, 1.0))
```

We now once again try a learning rate of 0.001, a batch size of 400, and 500 epochs. This yields 33.333% training accuracy and 33.333% validation accuracy, with the following loss graph:

![Loss Function Graph 19](./loss_function_graph_19.png)

Ok, the learning rate is definitely too low. In addition, we need to constrain the gradients more. Let's try clamping them to be from -0.1 to 0.1.

We will then try a learning rate of 0.01, a batch size of 400, and 500 epochs. This yields 33.333% training accuracy and 33.333% validation accuracy, with the following loss graph:

![Loss Function Graph 20](./loss_function_graph_20.png)

Definitely need to restrict the gradient more. We now try clamping them to be from -0.005 to 0.005.

We try the same parameters as before (a learning rate of 0.01, a batch size of 400, and 500 epochs). This yields 33.333% training accuracy and 33.332%, with the following loss graph:

![Loss Function Graph 21](./loss_function_graph_21.png)

Ok, perhaps the gradient clipping isn't working...I will test this on a subset, and get back to this.

After some testing, and another loss graph with a subset of the data (with a 0.01 learning rate, batch size of 400 and 500 epochs, with the gradient clamped to 1e-4):

![Loss Function Graph 22](./loss_function_graph_22.png)

Perhaps another way to reduce overfitting would be to go back to ShuffleNet, using L2 regularization to penalize large weights, and thus, force the model to do less overfitting. This time, however, we will use ShuffleNet 0.5x, just to ensure reducing overfitting will be easier with a smaller model (ShuffleNetV2 0.5x has about 1.4 million parameters [7]).

ShuffleNet

To start, we will get a baseline for what hyperparameters in general to use for training. We train, as we did with the last iteration of ShuffleNetV2, with a learning rate of 0.1, a batch size of 300, and 30 epochs. This yields 52.917% training accuracy and 38.0% validation accuracy, with the following loss graph:

![Loss Function Graph 23](./loss_function_graph_23.png)

Our loss is decreasing steadily once again, but it seems 30 epochs was not enough. In addition, the learning rate seems to be too low, as the reduction in loss was very slow.

No worries, we can simply increase the number of epochs (we will also slightly increase the learning rate to speed up training). We try a learning rate of 0.2, a batch size of 300, and 100 epochs. This yields 100.0% training accuracy and 65.875% validation accuracy, with the following loss graph:

![Loss Function Graph 24](./loss_function_graph_24.png)

At this point, we will now introduce L2 regularization through SGD's `weight_decay` parameter. To start, we will use a value of 1 for the regularization coefficient.

We train with a learning rate of 0.2, a batch size of 300, and 100 epochs. This yields 33.333% training accuracy and 33.333% validation accuracy, with the following loss graph:

![Loss Function Graph 25](./loss_function_graph_25.png)

It seems the coefficient was too high; the model is now underfitting. We will try several values of regularization coefficients, including 0.1, 0.01, and 0.001. 

We train with a learning rate of 0.2, a batch size of 300, and 100 epochs.

Regularization coefficient of 0.1: 33.333% training accuracy and 33.333% validation accuracy

![Loss Function Graph 26](./loss_function_graph_26.png)

Regularization coefficient of 0.01: 33.333% training accuracy and 33.333% validation accuracy

![Loss Function Graph 27](./loss_function_graph_27.png)

Regularization coefficient of 0.001: 33.333% training accuracy and 33.333% validation accuracy.

![Loss Function Graph 28](./loss_function_graph_28.png)

Perhaps the values are still too high; I will try 0.0001.

We train with a learning rate of 0.2, a batch size of 300, and 100 epochs. This yields 100.0% training accuracy and 63.875% validation accuracy, with the following loss graph:

![Loss Function Graph 29](./loss_function_graph_29.png)

Ok, now we're getting somewhere. We'll try values of 0.0002, 0.0004, 0.0006, and 0.0008.

We train with a learning rate of 0.2, a batch size of 300, and 100 epochs. 

Regularization coefficient of 0.0002: 100.0% training accuracy and 66.458% validation accuracy.

![Loss Function Graph 30](./loss_function_graph_30.png)

Regularization coefficient of 0.0004: 100.0% training accuracy and 67.75% validation accuracy.

![Loss Function Graph 31](./loss_function_graph_31.png)

Regularization coefficient of 0.0006: 100.0% training accuracy and 69.917% validation accuracy.

![Loss Function Graph 32](./loss_function_graph_32.png)

Regularization coefficient of 0.0008: 100.0% training accuracy and 67.5% validation accuracy.

![Loss Function Graph 33](./loss_function_graph_33.png)

We can probably decrease the epochs to 50 at this point, since we don't want to perfectly fit to the training data.

At this point, we try values of 0.00065, 0.0007 and 0.00075 for the regularization coefficient, with a learning rate of 0.2, a batch size of 300, and 50 epochs.

Regularization coefficient of 0.00065: 95.292% training accuracy and 60.0% validation accuracy.

![Loss Function Graph 34](./loss_function_graph_34.png)

Regularization coefficient of 0.0007: 94.403% training accuracy and 61.375% validation accuracy.

![Loss Function Graph 35](./loss_function_graph_35.png)

Regularization coefficient of 0.00075: 89.069% training accuracy and 56.458% validation accuracy.

![Loss Function Graph 36](./loss_function_graph_36.png)

Out of curiosity, we will try an experiment; is the model actually going to overfit when we train beyond 100% training accuracy?

We train with a learning rate of 0.2, a batch size of 300, and 250 epochs, with a regularization coefficient of 0.0006. This yields 100.0% training accuracy and 68.75% validation accuracy, with the following loss graph:

![Loss Function Graph 37](./loss_function_graph_37.png)

Okay, so it seems that more training can be a benefit. Though, to be safe, we will reduce the epochs to 200, since the validation accuracy did go slightly down.

We train with a regularization constant of 0.0005, learning rate of 0.2, batch size of 300, and 200 epochs. This yields 100.0% training accuracy and 66.542% validation accuracy, with the following loss graph:

![Loss Function Graph 38](./loss_function_graph_38.png)

Perhaps a higher coefficient with this amount of training would work better?

We train with a regularization constant of 0.005, learning rate of 0.2, batch size of 300, and 200 epochs. This yields 66.319% training accuracy and 51.416% validation accuracy, with the following loss graph:

![Loss Function Graph 39](./loss_function_graph_39.png)

Ok, the training and validation accuracies are getting closer. We now make a couple modifications; increase epochs to 500 and decrease learning rate to 0.02; these effects offset each other, and prevent the spike we saw on the right of the graph.

We train with a regularization constant of 0.005, learning rate of 0.02, batch size of 300, and 400 epochs. This yields 100.0% training accuracy and 42.958% validation accuracy, with the following loss graph:

![Loss Function Graph 40](./loss_function_graph_40.png)

Ok, it seems more regularization and epochs are needed. 

We train with a regularization constant of 0.01, learning rate of 0.02, batch size of 300, and 600 epochs. This yields 99.958% training accuracy and 72.583% validation accuracy, with the following loss graph:

![Loss Function Graph 41](./loss_function_graph_41.png)

Excellent, we're getting somewhere! Let's increase regularization, and slightly increase epochs. 

We train with a regularization constant of 0.02, learning rate of 0.02, batch size of 300, and 700 epochs. This yields 92.097% training accuracy and 62.292% validation accuracy, with the following loss graph:

![Loss Function Graph 42](./loss_function_graph_42.png)

Ok, we're getting some loss spikes; it seems the only way to reduce them is to increase the batch size [10]. In addition, to reduce overfitting and oscillation, we will lower the learning rate. In addition, we can decrease epochs by quite a bit; they don't seem to be doing much past around the 250 mark (but not by too much, since we have decreased the learning rate).

We train with a regularization constant of 0.02, learning rate of 0.002, batch size of 400, and 450 epochs. This yields 85.889% training accuracy and 26.25% validation accuracy, with the following loss graph:

![Loss Function Graph 43](./loss_function_graph_43.png)

Alright, learning rate is too low, and it seems like we need more regularization. Can add some epochs for good measure as well.

We train with a regularization constant of 0.05, learning rate of 0.008, batch size of 400, and 500 epochs. This yields 96.681% training accuracy and 66.208% validation accuracy, with the following loss graph:

![Loss Function Graph 44](./loss_function_graph_44.png)

Let's tune up the learning rate, epochs and regularization. At this point, if the coefficient gets to 0.5 and there is still overfitting, it will be time to introduce dropout.

We train with a regularization constant of 0.2, learning rate of 0.01, batch size of 400, and 600 epochs. This yields 33.333% training accuracy and 33.333% validation accuracy, with the following loss graph:

![Loss Function Graph 45](./loss_function_graph_45.png)

Ok, our regularization constant was probably too high. It also seems we need to reduce the learning rate when increasing regularization to ensure stability.

We train with a regularization constant of 0.1, learning rate of 0.006, batch size of 400, and 600 epochs. This yields 66.597% training accuracy and 49.917% validation accuracy, with the following loss graph:

![Loss Function Graph 46](./loss_function_graph_46.png)

Ok, the discrepancy between training and validation accuracy is improving, but it seems the learning rate is still too high. Let's decrease learning rate and increase regularization, also increasing batch size for good measure with respect to stability.

We train with a regularization constant of 0.2, learning rate of 0.001, batch size of 600, and 600 epochs. This yields 54.639% training accuracy and 29.833% validation accuracy, with the following loss graph:

![Loss Function Graph 47](./loss_function_graph_47.png)

Hm...perhaps using ShuffleNet 0.5x's pre-trained weights will work better? If this doesn't work, then it might be more fruitful to augument the dataset.

We train with pre-trained weights, a regularization constant of 0.0, learning rate of 0.001, batch size of 600, and 600 epochs. This yields 33.333% training accuracy and 33.333% validation accuracy, with the following loss graph:

![Loss Function Graph 48](./loss_function_graph_48.png)

Alright, definitely time to augument the dataset then. We will first try balancing the class distribution of the dataset (i.e. increasing the image counts for the 4 gestures from 2000 to 4000) by taking flips along the x axis for the 4 gesture classes. This increases our dataset to 20 000 images. 

We train with a learning rate of 0.1, a batch size of 300, and 50 epochs. This yields 93.042% training accuracy and 55.8% validation accuracy, with the following loss graph:

![Loss Function Graph 49](./loss_function_graph_49.png)

Alright, perhaps the model needs to train more?

We train with a learning rate of 0.1, a batch size of 300, and 200 epochs. This yields 100.0% training accuracy and 66.975% validation accuracy, with the following loss graph:

![Loss Function Graph 50](./loss_function_graph_50.png)

Ok, it seems that was effective; let's try more training.

We train with a learning rate of 0.1, a batch size of 300, and 400 epochs. This yields 100.0% training accuracy and 66.225% validation accuracy, with the following loss graph:

![Loss Function Graph 51](./loss_function_graph_51.png)

Alright, it may be time to expand the dataset; we will increase the size of the dataset in 2 steps; creating 8 progressively more blurred versions of each image using box blurring, increasing the dataset to 180 000 images. We will then flip all images along the y axis to double the samples. So our final image dataset now contains 360 000 images.

We train with a learning rate of 0.1, a batch size of 500, and 50 epochs. This yields 100.0% training accuracy and 99.968% validation accuracy, with the following loss graph:

![Loss Function Graph 52](./loss_function_graph_52.png)

Excellent, the model met (and even far exceeded our expectations). However, our job isn't quite done yet.

While yes, our model performed quite well, we still need to get deeper insight into its abilities (and ensure our results are reproducible). For this, we do a few things:

- as the model weights were iniialized randomly, set the random seed ahead of time to ensure the same results for the same hyperparameters.

- use a confusion matrix to test where the model is making mistakes (sure, the model's mistakes were few, but it's still interesting to see which classes it wasn't quite perfect on). This will also give us access to things like precision and recall.

Another thing we need to do is ensure there are no augmentations to the validation set (previously, I did augmentations on the validation and testing set, which I later learned is not advisable).

We train with a learning rate of 0.1, a batch size of 500, and 50 epochs. This yields 100.0% training accuracy and 32.0% validation accuracy, with the following loss graph:

![Loss Function Graph 53](./loss_function_graph_53.png)

Alright, let's decrease how much time the model trains for.

We train with a learning rate of 0.1, a batch size of 500, and 10 epochs. This yields 99.143% training accuracy and 27.6% validation accuracy, with the following loss graph:

![Loss Function Graph 54](./loss_function_graph_54.png)

Hm...I'll try removing the augmentation of varying degrees of blur. 

We train with a learning rate of 0.1, a batch size of 500, and 50 epochs. This yields 100.0% training accuracy and 23.0% validation accuracy, with the following loss graph:

![Loss Function Graph 55](./loss_function_graph_55.png)

Alright, perhaps we can try switching the model architecture to something even simpler. In addition, let's bring back the blurring augmentation; that seemed to be effective.

We switch to the LeNet-5 architecture [11], visualized below:

![LeNet-5 Picture](./lenet-5.png)

This architecture contains less parameters than ShuffleNet, however, we make two changes; we allow 3 input channels since our images are RGB images, and the first fully connected layer needs 2304 nodes rather than 400 to account for image size differences (to be more exact).

Using this new architecture, we train with a learning rate of 0.1, a batch size of 500, and 50 epochs. This yields 96.250% training accuracy and 31.8% validation accuracy, with the following loss graph:

![Loss Function Graph 56](./loss_function_graph_56.png)

Alright, perhaps the learning rate is too high. Let's decrease it.

We train with a learning rate of 0.01, a batch size of 500, and 100 epochs. This yields 93.522% training accuracy and 30.6% validation accuracy, with the following loss graph:

![Loss Function Graph 57](./loss_function_graph_57.png)

Let's try using the ShuffleNet architecture again, using our data augmentations. We also decrease the image size to 63x63 due to memory limitations on my local machine.

We train with a learning rate of 0.1, a batch size of 500, and 50 epochs, with an L2 regularization coefficient of 0.01. This yields 96.320% training accuracy and 53.2% validation accuracy, with the following loss graph:

![Loss Function Graph 58](./loss_function_graph_58.png)

The spikes are a good indication the learning rate is too high. Let's decrease it.

We train with a learning rate of 0.01, a batch size of 500, and 50 epochs, with an L2 regularization coefficient of 0.01. This yields 93.25% training accuracy and 45.6% validation accuracy, with the following loss graph:

![Loss Function Graph 59](./loss_function_graph_59.png)

Alright, an idea now would be to use a step-based learning rate scheduler to decrease the learning rate at 10 epoch intervals; we will use a gamma value of 0.1 for the scheduler (i.e. at every 10 epoch step the learning rate is multiplied by 0.1).

We train with a learning rate of 0.1, a batch size of 500, and 80 epochs, with an L2 regularization coefficient of 0.01, and using a step based learning rate scheduler with a gamma value of 0.1 and application every 10 epochs. This yields , with the following loss graph:

![Loss Function Graph 60](./loss_function_graph_60.png)

(In progress..)