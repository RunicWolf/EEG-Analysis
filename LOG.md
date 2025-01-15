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

