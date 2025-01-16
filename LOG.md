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


### Completed Tasks:
- **MLP Model Troubleshooting**:
  - Worked on debugging the MLP model. Fixed issues with target values during training.
  - Resolved class imbalance by applying class weighting.
  - Evaluated the model and analyzed metrics like accuracy, precision, recall, and F1 score.

- **Implemented CNN Model**:
  - Developed a Convolutional Neural Network (CNN) model for EEG data.
  - Addressed input tensor shape issues, successfully training the CNN model with the following results:
    - Epoch [150/150], Loss: 0.0283
    - Test Accuracy: 86.11%
  - Saved the trained CNN model and ensured correct evaluation.

- **Model Evaluation**:
  - After training the CNN model, the model was evaluated on the test set.
  - Achieved an accuracy of 86.11% on the test data.

- **Random Forest, XGBoost, and LightGBM Models**:
  - Trained multiple machine learning models: **Random Forest**, **XGBoost**, and **LightGBM**.
  - Performed hyperparameter tuning with **GridSearchCV** for Random Forest and **RandomizedSearchCV** for XGBoost and LightGBM.
  - Best results:
    - **Random Forest**: 
      - Best parameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
      - Best score: 91.77% accuracy.
    - **XGBoost**:
      - Best parameters: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 1.0}
      - Best score: 94.44% accuracy.
    - **LightGBM**:
      - Best parameters: {'num_leaves': 31, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.3}
      - Best score: 94.56% accuracy.
  - Evaluated all three models, and test accuracies:
    - **Optimized Random Forest Test Accuracy**: 90.56%
    - **Optimized XGBoost Test Accuracy**: 94.44%
    - **Optimized LightGBM Test Accuracy**: 92.78%

- **Logging Progress**:
  - Updated `LOG.md` with the results of the MLP model, CNN model, Random Forest, XGBoost, and LightGBM models.
  - Logged the challenges faced, such as tensor shape mismatches, class imbalance in MLP, and tuning machine learning models.

### Issues Resolved:
- Fixed tensor shape mismatch issues during CNN model training.
- Resolved problems with class weighting in the MLP model due to class imbalance in the dataset.
- Successfully handled the training and evaluation of the CNN model on EEG data.
- Addressed hyperparameter tuning for Random Forest, XGBoost, and LightGBM models.

### Next Steps:
- Continue with the model evaluation for more detailed insights and comparison with other models.
- Experiment with LSTM or other deep learning architectures for EEG data analysis.
- Fine-tune hyperparameters for optimal performance.

### Achievements:
- Successfully trained CNN, Random Forest, XGBoost, and LightGBM models and achieved satisfactory results in EEG analysis.
- Models saved and ready for further evaluation or deployment.
