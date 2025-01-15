# Project Log

This file contains a detailed log of all changes and milestones for the `EEG-Analysis-Chatbot` project.

---

## **January 2025**

### **15th January 2025**
- Implemented preprocessing pipeline:
  - Handled missing values using interpolation.
  - Removed outliers using the z-score method with a threshold of 3.
  - Normalized data using z-score normalization.
  - Extracted features (mean, variance, skewness, kurtosis, PSD).
  - Split data into training (70%), validation (15%), and test (15%) sets with stratification.
- Saved processed datasets as:
  - `train.pt`
  - `val.pt`
  - `test.pt`
- Logged progress and changes on GitHub.

---

## **Upcoming Work**
- Build the model training pipeline.
- Explore hyperparameter tuning for optimal performance.

### **16th January 2025**
- Successfully balanced the dataset:
  - Original label distribution: `(array([0.], dtype=float32), array([50]))`
  - Balanced label distribution: `(array([0, 1, 2]), array([80, 80, 80]))`
  - Synthetic classes created using data augmentation (Gaussian noise, amplitude scaling).
- Applied PCA for dimensionality reduction:
  - Explained variance ratio: `[0.9704, 0.0223, 0.0070, 0.0002]`.
- Selected top features using `SelectKBest`:
  - Valid feature scores: `[52.03, 32.14, 77.79, 17.77]`.
- Split the dataset:
  - Training, validation, and test sets saved as `.pt` files.
- Files saved in: `C:\Users\prajw\Desktop\docs\All Projects\EEG\EEG-Analysis-Chatbot\processed_data\Bon_UKB`.

