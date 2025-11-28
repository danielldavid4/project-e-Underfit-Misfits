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

