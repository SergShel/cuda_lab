<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="Assignment-8">Assignment 8<a class="anchor-link" href="#Assignment-8">¶</a></h1><h4 id="Note:-The-CudaLab-project-is-an-extension-of-this-assignment.-I-recommend-reaching-a-good-solution-in-this-assignment,and-then-extend-the-model-as-required-in-the-project."><strong>Note:</strong> The CudaLab project is an extension of this assignment. I recommend reaching a good solution in this assignment,and then extend the model as required in the project.<a class="anchor-link" href="#Note:-The-CudaLab-project-is-an-extension-of-this-assignment.-I-recommend-reaching-a-good-solution-in-this-assignment,and-then-extend-the-model-as-required-in-the-project.">¶</a></h4><ul>
<li>Implement a UNet model:<ul>
<li>Choose your own encoder and decoder</li>
<li>Residual connections</li>
</ul>
</li>
<li>Train your model on the Cityscapes dataset<ul>
<li>Autonomous driving scenes</li>
<li>Not too much data:<ul>
<li>Data augmentation</li>
<li>Transfer learning</li>
<li>Loss weighting</li>
<li>...</li>
</ul>
</li>
</ul>
</li>
<li>Evaluate your model using Accuracy, mIoU and Dice Coefficient</li>
<li>Show some images and the results</li>
<li>Which classes are the most problematic?</li>
<li><strong>Extra Point</strong>: Train the same model as before, but with the following loss functions, and compare the results:<ul>
<li>Dice Loss</li>
<li>Focal Loss</li>
</ul>
</li>
</ul>

</div>
<br>

Remarks to the solution:

* assignment_8.ipynb contains the main part of the solution with the pretrained ResNet backbone. 
* test_convUnext.ipynb contains similar solution, but with the ConvNeXt-like backbone (heavily inspired by: https://github.com/1914669687/ConvUNeXt/blob/master/src/ConvUNeXt.py)
* According to the metrics (mIoU, Dice Loss and Focal Loss) and visual evaluation the U-Net with ConvNext (ConvUNeXt) performs better than U-Net with the pretrained ResNet backbone.
* As the Dataset is unbalanced for some classes (wall, fence, bus and train) we have applied CopyBlob augmentation (https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/learning/utils.py)
<br> Here is an example of copyBlob augmentation:
<img src="https://github.com/SergShel/cudaLab/blob/main/Assignment_8/resources/copyBlobSample.png">
