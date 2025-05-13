## Structure

### augmentation/
- **`cyclegan_resnet.py`** - Implements a CycleGAN model using a ResNet-based generator for realistic skin tone transformations.
- **`cyclegan.py`** - Implements a CycleGAN model using a U-net-based generator for realistic skin tone transformations.
- **`data_loading.py`** - Handles loading and preprocessing for the CycleGAN models.
- **`signal_processing.py`** - Applies signal processing techniques for skin tone transformations.
- **`training_res_cgan.py`** - Training script for CycleGAN model with ResNet-based generator.
- **`training.py`** - Training script for CycleGAN model with U-net-based generator.

---

### classification/
- **`data_loading.py`** - Contains loading and preprocessing functions of skin lesion datasets for classification tasks.
- **`evaluate.py`** - Contains functions for evaluating model performance.
- **`models.py`** - Defines (pretrained) deep learning models.
- **`plot.py`** - Generates plots for training curves, precision-recall, and ROC curves.
- **`train.py`** - Contains training functions for classification models.
- **`main_1.py`** to **`main_8.py`** - Individual training scripts for different classification experiments.