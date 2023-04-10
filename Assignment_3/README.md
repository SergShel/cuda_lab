### Assignment 3

#### Car-Type Classiciation on a 'small' dataset:
- Stanford Cars dataset: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- 16.185 images
- 196 classes
- On average, 90 images per class

#### Augmentations

For your experiments, use augmentations from the following types:
<ul>
  <li>Spatial Augmentations (rotation, mirroring, croppoing, ...)</li>
  <li>Use some other augmentations (color jitter, gaussian noise, ...).</li>
  <li>
    Use one (or more) of the following advanced augmentations:
    <ul>
       <li>CutMix: https://arxiv.org/pdf/1905.04899.pdf</li>
      <li>Mixup: https://arxiv.org/pdf/1710.09412.pdf</li>
    </ul>
  </li>
</ul>

#### Experiments:

Using your aforementioned augmentions:

<ol>
  <li>Fine-tune VGG, ResNet and ConvNext for your augmented dataset for car type classification and compare them.</li>
  <li>Compare the following: Fine-Tuned ResNet, ResNet as fixed feature extractor, and ResNet with a Combined Approach</li>
  <li>Log your losses and accuracies into Tensorboard (or some other logging tool)</li>  
  <li>Bonus task:</li>
  * Fine-tune a Transformer-based model (e.g. ViT). Compare the performance (accuracy, confusion matrix, training time, loss landscape, ...) with the one from ResNet.
</ol>

<br>

- assignment_3.ipynb - contains my solution for this Assignment. <br>
