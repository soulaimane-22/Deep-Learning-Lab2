# Deep Learning Lab 2

Classifying MNIST digits with PyTorch on GPU using a CNN, Faster R-CNN, pretrained VGG16 and AlexNet, and a Vision Transformer built from scratch.

All experiments run in Kaggle. The repository stores notebooks and selected figures; raw datasets, model checkpoints, and detailed local report exports are excluded.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [CNN](notebooks/01_cnn_mnist.ipynb) | Convolution, pooling, and fully connected layers |
| [Faster R-CNN](notebooks/02_faster_rcnn_mnist.ipynb) | Detection adaptation and comparison with the CNN |
| [VGG16](notebooks/03_vgg16_mnist.ipynb) | Fine-tuning pretrained VGG16 |
| [AlexNet](notebooks/04_alexnet_mnist.ipynb) | Fine-tuning pretrained AlexNet |
| [Vision Transformer](notebooks/05_vit_mnist.ipynb) | ViT implementation and training from scratch |

## Run in Kaggle

1. Import the CNN notebook from this repository.
2. Attach the [MNIST dataset specified in the assignment](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and enable a GPU.
3. Execute the setup cells in order and review their outputs before starting model training.
4. Download the executed notebook to the matching local `notebooks/` path. Copy selected figures from `/kaggle/working/plots/` to `assets/plots/`. Keep detailed exports from `/kaggle/working/results/` in local, Git-ignored `reports/`.

The notebooks record installed package versions and GPU information. Their setup cells share the same split procedure and print a split hash that must match across experiments.

## Data and comparison protocol

The expected official MNIST partitions contain 60,000 training and 10,000 test images. The setup checks these sizes, then reserves 6,000 official training images for stratified validation using seed 42. This gives 54,000 training, 6,000 validation, and 10,000 test images for every model.

Model and checkpoint choices use validation only. The final test comparison is performed after all configurations are frozen. We report accuracy, macro-F1 across ten digits, loss, and training time. GPU, epochs, batch size, transformations, and pretrained initialization are documented for each experiment.

For Faster R-CNN, the proposed adaptation assigns one full-image box to each digit. Detector label 0 is reserved for background. The highest-scoring detection provides the digit prediction; missing detections count as errors. These boxes are derived annotations, not localization labels supplied by MNIST. Detection loss is reported separately from classifier cross-entropy because their objectives differ.

VGG16 and AlexNet use pretrained weights and model-specific input transforms. The CNN and ViT are trained from scratch. Training time includes per-epoch validation and excludes downloads, with CUDA synchronization around timing.

## Progress and results

The repository structure and Issue 1 setup cells are prepared. Kaggle execution is pending; no experimental results are claimed yet. The final report will summarize both parts, compare all five models, show significant figures, and include a brief personal synthesis.

The eight [GitHub issues](https://github.com/soulaimane-22/Deep-Learning-Lab2/issues) track the roadmap: setup, CNN, Faster R-CNN, their comparison, VGG16, AlexNet, scratch ViT, and the final report.

## References

- The professor's Lab 2 assignment, supplied separately.
- [Assigned ViT tutorial](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c).
- [Torchvision Faster R-CNN documentation](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html).
