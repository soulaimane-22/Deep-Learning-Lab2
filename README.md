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

The Kaggle setup verified 60,000 official training images and 10,000 test images, each measuring 28 by 28 pixels. A stratified split with seed 42 reserves 6,000 training images for validation. Both executed notebooks use the same 54,000 training, 6,000 validation, and 10,000 test image partitions. The matching split hash confirms identical source indices.

The setup also verified CUDA execution, finite gradients, valid labels, and coverage of all ten digits. Training class counts range from 4,879 to 6,068, so no resampling was applied. The environment used Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and Tesla T4 GPUs.

Model and checkpoint choices use validation only. The final test comparison is performed after all configurations are frozen. We report accuracy, macro-F1 across ten digits, loss, and training time. GPU, epochs, batch size, transformations, and pretrained initialization are documented for each experiment.

For Faster R-CNN, the proposed adaptation assigns one full-image box to each digit. Detector label 0 is reserved for background. The highest-scoring detection provides the digit prediction; missing detections count as errors. These boxes are derived annotations, not localization labels supplied by MNIST. Detection loss is reported separately from classifier cross-entropy because their objectives differ.

VGG16 and AlexNet use pretrained weights and model-specific input transforms. The CNN and ViT are trained from scratch. Training time includes per-epoch validation and excludes downloads, with CUDA synchronization around timing.

## Progress and results

### CNN

The CNN uses two convolution blocks with 32 and 64 channels. Each block uses a 3-by-3 kernel, stride 1, padding 1, ReLU, and 2-by-2 max pooling. A 128-unit fully connected layer, ReLU, dropout of 0.25, and a ten-logit output form the classifier. The model has 421,642 trainable parameters and was trained from scratch with Adam, learning rate 0.001, batch size 64, and unweighted cross-entropy for ten epochs.

Epoch 10 was selected by the lowest validation loss. It achieved 99.15% validation accuracy and 0.991446 macro-F1, with 51 errors among 6,000 images. Training accuracy was 99.85%, a gap of 0.70 percentage points. Every digit achieved an F1 score above 0.98; digits 8 and 9 had the lowest F1 scores.

![CNN training and validation curves](assets/plots/cnn_learning_curves.png)

Validation loss fluctuated after the first few epochs but reached its minimum at epoch 10. These results describe validation performance; the test set has not been evaluated.

### Faster R-CNN

The full detector uses a small custom backbone trained from scratch, a region proposal network, RoI pooling, and classification and box-regression heads. It has 484,388 trainable parameters. Ten epochs of Adam training used learning rate 0.001 and a global batch size of 32, split across two T4 GPUs with DistributedDataParallel. Digit labels 0-9 map to detector labels 1-10; background occupies label 0. Inference retains the highest-scoring detection with a fixed score threshold of 0.0.

Epoch 9 was selected by the highest validation macro-F1. Its restored weights independently reproduced 98.8833% validation accuracy and 0.988829 macro-F1, with 67 errors and no missing detections. Digit 9 had the lowest recall, followed by digit 3. The derived full-image boxes adapt the detector to digit classification; they do not establish performance on a genuine localization task.

![Faster R-CNN training and validation curves](assets/plots/faster_rcnn_learning_curves.png)

Epoch 10 had a lower detection loss but lower validation macro-F1. Detection loss combines several objectives, so it need not decrease in step with classification errors.

### Recorded validation results

| Model | Selected epoch | Accuracy | Macro-F1 | Loss | Loss type | Run time | GPUs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 10 | 99.15% | 0.991446 | 0.035260 | Cross-entropy | 32.10 seconds | 1 T4 |
| Faster R-CNN | 9 | 98.8833% | 0.988829 | 0.034690 | Total detection loss | 57.20 minutes | 2 T4s |

Times include full training and validation evaluations after each epoch. The detector also performs separate inference and loss passes and includes checkpoint/history writes. These are measured run times under different workloads and GPU counts, not a controlled hardware speed comparison. The two loss values measure different objectives and must not be ranked against each other. CNN checkpoints were selected by validation loss, while detector checkpoints were selected by validation macro-F1.

Both models saved and restored their selected checkpoints in Kaggle. Checkpoint files are not currently available in the local export. The CNN export was deferred at the user's request; recovery of the selected weights, or a documented reproducible retraining, is required before final test evaluation. Reports and any recovered checkpoints remain local and excluded from Git.

The formal CNN/Faster R-CNN comparison, transfer-learning experiments, scratch ViT, final test evaluation, and personal synthesis remain to be completed.

The eight [GitHub issues](https://github.com/soulaimane-22/Deep-Learning-Lab2/issues) track the roadmap: setup, CNN, Faster R-CNN, their comparison, VGG16, AlexNet, scratch ViT, and the final report.

## References

- The professor's Lab 2 assignment, supplied separately.
- [Assigned ViT tutorial](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c).
- [Torchvision Faster R-CNN documentation](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html).
