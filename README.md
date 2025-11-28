# Project E – Simple Object Tracking

This project implements a simple convolutional neural network (CNN) that performs:

- **Per-frame object classification** (ball, mug, pen, spoon, notebook)
- **Per-frame bounding-box regression** for the visible object

The model is trained from scratch on the provided project dataset and evaluated using:

- Frame-level **classification accuracy**
- **Average Intersection-over-Union (IoU)** over correctly classified frames

The code is organized into separate training and testing notebooks to make it easy to run and grade.

---

## Repository Structure

```text
project-e-Underfit-Misfits/
│
├── data/
│   ├── training_data_projectE.npy
│   ├── training_labels_projectE.npy
│   └── (blind test .npy will be placed here or elsewhere by the user)
│
├── models/
│   └── simple_cnn_class_and_bbox.h5      # trained model weights
│
├── notebooks/
│   ├── 02_train_cnn.ipynb                # trains the CNN
│   └── 03_test_cnn.ipynb                 # evaluates + runs on blind test set
│
├── Final_Project_csv/                    # (or external path; bounding-box CSVs)
│   └── *.csv                             # one CSV per annotated training clip
│
├── README.md
└── .gitignore
```

## Dependencies

- numpy>=1.23
- pandas>=1.5
- matplotlib>=3.6
- tensorflow>=2.10
- h5py>=3.7
- scikit-learn>=1.2

## Data Description

### **Training Data (.npy)**  
Located in the `data/` directory:

- **`training_data_projectE.npy`**  
  Shape: `(num_clips, 15, 100, 100, 3)`  
  Contains 15 RGB frames (100×100) per clip.

- **`training_labels_projectE.npy`**  
  Shape: `(num_clips, 15, 5)`  
  One-hot encoded labels for 5 object classes.

### **Bounding Box Data (.csv)**  
A set of CSV files stored in the `Final_Project_csv/` folder.  
Each CSV corresponds to one annotated clip and contains:

- `bbox_x`, `bbox_y`
- `bbox_width`, `bbox_height`
- `image_width`, `image_height`

Bounding boxes are converted into normalized coordinates:

### **Blind Test Data (.npy)**  
Provided by the instructor:  

No labels or bounding boxes are included.  
Your model must produce class and bounding-box predictions for all frames.

## Model Architecture

A compact, two-head convolutional neural network (CNN) is used.

### **Backbone (Shared Feature Extractor)**
- Conv2D → ReLU → MaxPooling  
- Conv2D → ReLU → MaxPooling  
(standard 2-block CNN)

### **Classification Head**
- Dense layer(s)
- Softmax output with 5 classes
- Loss: `sparse_categorical_crossentropy`

### **Bounding Box Regression Head**
- Dense layer(s)
- Linear output of 4 values: `[x_min, y_min, x_max, y_max]`
- Loss: Huber loss

### **Loss Weighting**
```python
loss_weights = {
    "class_head": 1.0,
    "bbox_head": 5.0
}
```

---

# Training Instructions

## Training Instructions

Training is performed in the notebook:


Run all cells in order. The notebook performs:

1. **Load the training `.npy` data**
2. **Load bounding-box CSVs** via `csv_root`
3. **Convert clips → frame-level dataset**
4. **Define the CNN (shared backbone + two heads)**
5. **Compile and train the model**
6. **Save trained model** to:

After training completes, the saved model is ready for evaluation and blind test prediction.

## Testing Instructions

Testing is performed in:

This notebook:

1. **Loads the trained model** from `models/`
2. **Rebuilds the annotated dataset**  
   using `.npy` files and CSV bounding boxes
3. **Computes evaluation metrics**, including:
   - classification accuracy  
   - classification loss  
   - bounding-box regression loss  
   - **Average IoU (Intersection-over-Union)** over correctly classified frames

These metrics are printed directly in the notebook for grading.

## Running Blind Test Predictions

The instructor provides a blind test `.npy` file with shape:

To evaluate it:

1. Open `notebooks/03_test_cnn.ipynb`
2. Find the cell labeled **"Provide Blind Test File Path"**
3. Edit this line:

```python
blind_test_path = r"C:/path/to/blind_test.npy"

blind_test_classes.npy   # predicted class labels per frame
blind_test_bboxes.npy    # predicted bounding boxes per frame
```

---

# Adjustable Parameters

## Adjustable Parameters

All adjustable parameters are located at the top of the training and testing notebooks.

### **Training Parameters**
- `num_epochs`  
- `batch_size`  
- `learning_rate`

### **Loss Weights**
```python
loss_weights = {
    "class_head": 1.0,
    "bbox_head": 5.0
}
```

---

# Reproducibility

## Reproducibility

To reproduce the results:

1. Install dependencies using `pip install -r requirements.txt`
2. Place `.npy` data files inside the `/data` folder
3. Place bounding-box CSVs in the folder referenced by `csv_root`
4. Run all cells in:
   - `notebooks/02_train_cnn.ipynb`
   - `notebooks/03_test_cnn.ipynb`
5. Provide the path to the blind test `.npy` in the indicated cell
6. Generated prediction files (`*_classes.npy`, `*_bboxes.npy`) can be submitted directly

The entire codebase runs with no edits except for path configuration.

## Summary

This project trains and evaluates a lightweight CNN for object classification and bounding-box regression using short video clips. The solution includes:

- A clean dataset-loading pipeline
- A compact two-head CNN architecture  
  (classification + bounding box regression)
- Evaluation of classification accuracy and IoU
- An easy-to-run blind test prediction workflow
- Adjustable model and training parameters
- Fully reproducible results with minimal user setup

The code is organized cleanly, documented thoroughly, and designed to run smoothly in a grading environment.
