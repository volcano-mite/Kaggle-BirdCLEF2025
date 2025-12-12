# BirdCLEF 2025 - Bird Sound Recognition

## Project Overview

This repository contains training and inference code for the [BirdCLEF 2025](https://www.kaggle.com/competitions/birdclef-2025) competition on Kaggle. The task is to identify bird species from audio recordings using deep learning techniques.
You can check it on the kaggle:
Model train: https://www.kaggle.com/code/licanhou/model-train
Inference: https://www.kaggle.com/code/licanhou/inference

dataset:
training data: https://www.kaggle.com/competitions/birdclef-2025/data
Mel-Spectrom: https://www.kaggle.com/datasets/kadircandrisolu/birdclef25-mel-spectrograms
Test data is unavailable on Kaggle. And you can only see the test score once you submit.


### Model Architecture
- **Backbone**: EfficientNet-B0
- **Input**: Mel-Spectrogram (256×256, single channel)
- **Audio Processing**: 5-second audio clips at 32kHz sample rate
- **Output**: Multi-label classification for 206 bird species

### Key Features
- Pre-computed spectrogram support for faster training
- Data augmentation (SpecAugment, Mixup)
- Single-split validation strategy
- Temporal smoothing for inference predictions

---

## Requirements

```
Python 3.11+
torch
torchvision
timm
librosa
soundfile
opencv-python
numpy
pandas
scikit-learn
tqdm
```

For Kaggle environment, these dependencies are pre-installed.

---

## Data Preparation

### Required Data Structure
```
birdclef-2025/
├── train_audio/          # Training audio files (.ogg)
├── test_soundscapes/     # Test audio files (.ogg)
├── train.csv            # Training metadata
├── taxonomy.csv         # Species taxonomy
└── sample_submission.csv
```

### Optional: Pre-computed Spectrograms
For faster training, you can use pre-computed mel spectrograms:
- Set `CFG.spectrogram_npy` path to your `.npy` file
- Format: Dictionary with keys as `{species}-{filename}` and values as (256, 256) numpy arrays
- If not available, spectrograms will be computed on-the-fly (slower)

---

## Training

### Configuration
Edit the `CFG` class in `model_train.ipynb`:

```python
class CFG:
    seed = 42
    debug = False  # Set True for quick testing
    
    # Paths
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    output_dir = '/kaggle/working/model'  # Model save location
    
    # Training hyperparameters
    epochs = 10
    batch_size = 32
    lr = 5e-4
    val_pct = 0.2  # Validation split ratio
    
    # Audio parameters
    SR = 32000
    target_duration = 5
    n_mels = 128
    target_shape = (256, 256)
```

### Run Training

1. **On Kaggle**:
   - Create a new notebook
   - Add BirdCLEF 2025 competition data
   - (Optional) Add pre-computed spectrograms dataset
   - Copy the code from `model_train.ipynb`
   - Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
   - Run all cells

2. **Locally**:
   ```bash
   jupyter notebook model_train.ipynb
   ```

### Training Process
The training script will:
1. Load training metadata and taxonomy
2. Create train/validation split (80/20 by default)
3. Load or compute mel spectrograms
4. Train EfficientNet-B0 model with:
   - BCEWithLogitsLoss (multi-label classification)
   - AdamW optimizer
   - OneCycleLR scheduler
   - Mixup augmentation
5. Save best model based on validation AUC

### Training Output
- Best model saved to: `/kaggle/working/model/best_model.pth`
- Contains: `model_state_dict` and `cfg` configuration
- Evaluation metric: ROC-AUC score
- Training typically takes 20-30 minutes with GPU

---

## Inference

### Configuration
Update model path in `inference.ipynb`:

```python
class CFG:
    # Model path - IMPORTANT: Upload your trained model to a Kaggle dataset
    model_files = [
        '/kaggle/input/your-model-dataset/best_model.pth'
    ]
    
    # Must match training settings exactly
    model_name = 'efficientnet_b0'
    SR = 32000
    target_duration = 5
    n_mels = 128
    target_shape = (256, 256)
    n_fft = 1024
    hop_length = 512
    f_min = 50
    f_max = 14000
```

### Run Inference

1. **Upload Model**:
   - Download `best_model.pth` from training output
   - Go to Kaggle → Your Work → Datasets → New Dataset
   - Upload the model file
   - Make dataset public or add to notebook

2. **Create Submission Notebook**:
   - Create new notebook in BirdCLEF 2025 competition
   - Add your model dataset as input
   - Copy code from `inference.ipynb`
   - Enable GPU (optional but faster)

3. **Execute**:
   - Run all cells
   - Output: `submission.csv` in `/kaggle/working/`
   - Submit to competition

### Inference Pipeline
1. **Load Model**: Load trained weights from checkpoint
2. **Process Audio Files**:
   - Read test soundscapes (.ogg files)
   - Split long recordings into 5-second segments
   - Generate mel spectrograms for each segment
3. **Prediction**:
   - Batch inference on all segments
   - Apply sigmoid to get probabilities
4. **Post-processing**:
   - Temporal smoothing across consecutive segments
   - Format as competition submission
5. **Output**: CSV with predictions for each 5-second interval

---

## File Structure

```
.
├── model_train.ipynb      # Training pipeline
├── inference.ipynb        # Inference pipeline
├── README.md             # This file
└── outputs/
    ├── best_model.pth    # Trained model weights
    └── submission.csv    # Competition submission
```

-
### Performance Tips

1. **Pre-compute Spectrograms** 
   - Saves 10x training time
   - Use batch processing scripts
   - Store as `.npy` dictionary

2. **Hardware Optimization**
   - Enable GPU on Kaggle (T4 x2 recommended)
   - Use mixed precision training (`model.half()`)
   - Increase `num_workers` for faster data loading

3. **Model Ensemble**
   - Train multiple models with different:
     - Random seeds
     - Architectures (EfficientNet-B1, B2)
     - Audio parameters
   - Average predictions for better scores

4. **Hyperparameter Tuning**
   - Experiment with learning rates
   - Try different augmentation strategies
   - Adjust temporal smoothing weights
   - Test different backbone architectures

5. **Advanced Techniques**
   - Pseudo-labeling with test data
   - Cross-validation for better generalization
   - Focal loss for class imbalance
   - Attention mechanisms

---

## Competition Tips

### Evaluation Metric
- **Metric**: Macro-averaged ROC-AUC
- Each bird species gets equal weight
- Predictions are probabilities [0, 1]

### Submission Format
```csv
row_id,species1,species2,...,species206
soundscape_123_5,0.1,0.0,...,0.8
soundscape_123_10,0.2,0.1,...,0.3
```

### Training Time
- Single model: 1-2 hours (Kaggle GPU)
- Full pipeline: 2-3 hours including data prep

---



