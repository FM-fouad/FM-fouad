
Transfert Learning:
This tutorial includes:
1. Read data 
2. Data augmentation 
   In order to be able to be used on new data, our model of Deep Learning must not have been specialized during its training (over-fitting). On the contrary, he must have learned to generalize.
   a) For this it is necessary that the space of the characteristics of the learning data also covers the spectrum of possibilities, i.e they contain the largest possible number of representations of each characteristic of the data.
   b) Data Augmentation techniques includes affine transformations such as flip, the rotation, and zoom.
3. Compare GlobalAveragePooling and Flatten 
    a)Validation accuracy is approximately one (>99.5%)in the fully-connected layers model.
    b)The model converges in the GlobalAveragePooling model more smoothly than the previous one but with validation accuracy 98%.
4. & 5. Transfer learning
    a)In the first part, we take the Inception pretrained model on Imagenet, we make these layers untrainable, we eliminate its fully connected layers and we add another fully connected layer more adapted to our 17 classes's classification, we obtained good results for both training and validation
    b) In the second part, we take the same model as before but now we make one example with 172 layers untrainable and another one with 120 untrainable layers, and the rest are trainable, we obtained good results for both training and validation
    c) In the third part, we take another model, which Mobile net, with less layers than Inception, we freezed 28 layers, and the rest are trainable and  also we obtained good results for both training and validation

6. Conclusion
    a) More layers are trainable, better results are obtained
    b) Pretrained mode takes less time from untrained model and give nice results
    c) In each part, we made plot for accuracies and losses for training and validation to give us an overview about  how our model is trained and if it suffered of overfitting with other problems or not
    d) We have also visualization where we have images and their path which indicates the true class and in the second line we print the estimated class 
