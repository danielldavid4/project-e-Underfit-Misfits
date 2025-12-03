# Project E – CoordConv-Enhanced Object Classification and Localization

This project implements a multi-task deep learning model that performs:

- Object classification across five categories (Ball, Mug, Pen, Spoon, Notebook)
- Bounding-box regression to localize the object within each frame

The architecture uses a U-Net encoder–decoder, residual blocks, and a CoordConv-based bounding-box head, enabling strong spatial reasoning and high accuracy. Training and testing are cleanly separated into two notebooks.

---

# 1. Repository Structure

```
project-e-Underfit-Misfits/
│
├── data/
│   ├── training_data_projectE.npy
│   ├── training_labels_projectE.npy
│   └── (blind test .npy placed here)
│
├── models/
│   ├── best_model_coord.keras
│   └── final_model_coord.keras
│
├── notebooks/
│   ├── 02_train_cnn.ipynb
│   └── 03_test_cnn.ipynb
│
├── README.md
└── .gitignore
```

---

# 2. Dependencies

Recommended packages:

- numpy  
- matplotlib  
- tensorflow >= 2.10  
- scikit-learn  
- h5py  

---

# 3. Data Description

## 3.1 Training Data (.npy)

### training_data_projectE.npy
Shape:
```
(num_clips, 15, 100, 100, 3)
```
Each clip contains 15 RGB frames at 100×100 resolution.

### training_labels_projectE.npy
Shape:
```
(num_clips, 15, 5)
```
Each frame contains:
```
[class_id, bbox_x, bbox_y, bbox_width, bbox_height]
```

Bounding boxes are normalized to the range [0, 1].

---

## 3.2 Blind Test Data

Contains the same structure as the training data but without labels.

The model must output:

- predicted_classes.npy  
- predicted_bboxes.npy (normalized x, y, w, h)

---

# 4. Model Architecture

A two-head convolutional neural network designed for spatial reasoning.

## 4.1 Encoder (Backbone)

A series of residual blocks with:

- 32 filters  
- 64 filters  
- 128 filters  
- 256 filters  

Each block includes Conv2D, BatchNorm, LeakyReLU, and a skip connection.

## 4.2 Decoder (U-Net Style)

- Upsampling layers  
- Skip connections from encoder  
- Residual refinement blocks  

## 4.3 Classification Head

- Global Average Pooling  
- Dropout  
- Dense softmax (5 classes)  
- Loss: sparse categorical cross-entropy  

## 4.4 Bounding Box Regression Head (CoordConv)

- AddCoords layer adding (x, y) coordinate channels  
- Strided convolution layers  
- Dense sigmoid output for [x, y, w, h]  
- Loss: MAE, weighted ×10  

## 4.5 UNKNOWN Class (Optional)

If:
```
max(softmax_probs) < threshold
```
the model outputs UNKNOWN.

---

# 5. Training Instructions

Training is done in:

## notebooks/02_train_cnn.ipynb

This notebook:

1. Loads `.npy` training data  
2. Flattens clips into frame-level samples  
3. Removes invalid bounding boxes  
4. Applies augmentation  
5. Builds the CoordConv U-Net model  
6. Trains using:
   - Cosine-decay learning rate  
   - Adam optimizer  
   - Multi-task loss  
   - Early stopping  
7. Saves:
   - best_model_coord.keras  
   - final_model_coord.keras  

Plots for accuracy and loss are automatically generated.

---

# 6. Testing Instructions

Testing is done in:

## notebooks/03_test_cnn.ipynb

This notebook:

1. Loads the saved model  
2. Loads evaluation or blind test data  
3. Computes:
   - Classification accuracy  
   - Precision, recall, F1-score  
   - Mean IoU  
4. Visualizes predictions (ground truth = green, prediction = red)  
5. Saves blind test predictions  

---

# 7. Blind Test Workflow

Steps:

1. Open `03_test_cnn.ipynb`  
2. Insert the blind test `.npy` file path  
3. Run all notebook cells  

Outputs:

- blind_test_pred_classes.npy  
- blind_test_pred_bboxes.npy  

---

# 8. Adjustable Parameters

Found at the top of both notebooks:

- Learning rate  
- Batch size  
- Number of epochs  
- Loss weights  
- Augmentation settings  
- UNKNOWN threshold  

---

# 9. Reproducibility

To reproduce:

1. Install dependencies  
2. Place `.npy` files in the `data/` directory  
3. Run:
   - 02_train_cnn.ipynb  
   - 03_test_cnn.ipynb  
4. Provide blind test path when prompted  
5. Submit prediction `.npy` files  

No modification to model code is required beyond path updates.

---

# 10. Summary

This project uses a CoordConv-enhanced U-Net for simultaneous object classification and bounding-box prediction. Residual blocks, multi-scale features, and coordinate-aware bounding boxes provide strong generalization and high accuracy.

Performance highlights:

- ~92% classification accuracy  
- Strong bounding-box IoU  
- Robust predictions on unseen data  

The workflow is fully reproducible and structured for easy viewing.
