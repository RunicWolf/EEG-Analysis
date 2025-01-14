# Project Log

## January 10, 2025
- Downloaded datasets: CHB-MIT, Bonn, Sleep-EDF, BCI Competition IV.

## January 11, 2025

### Summary of Work:
1. **Preprocessing Script Updates for CHB-MIT:**
   - Implemented and tested a complete preprocessing pipeline for EEG `.edf` files.
   - The pipeline:
     - Processes `.edf` files to `.fif` format using MNE.
     - Applies GPU-accelerated bandpass filtering (0.5–50 Hz).
     - Adds seizure annotations from the corresponding summary files.
     - Handles duplicate channel names gracefully.
   - Verified seizure annotations for key files (e.g., `chb01_03.edf`).

2. **Verification of Output:**
   - Checked `processed_data` directory for generated `.fif` files.
   - Verified annotations using MNE’s API:
     - Files with seizures include correct start and end times.
     - Files without seizures contain zero annotations.
   - Visualized EEG signals with annotations using `raw.plot()`.

3. **Debugging and Logging Improvements:**
   - Added logs for summary file detection, seizure times, and processing progress.
   - Suppressed warnings about duplicate channel names.
   - Handled missing files and errors gracefully.

### Accomplishments:
- Processed all `.edf` files in the `chb01` dataset to `.fif` format.
- Ensured seizure annotations align with summary file data.
- Validated the preprocessing pipeline with clean outputs and accurate logs.

## January 12, 2025

### Summary of Work:
1. **Sleep-EDF Dataset Preprocessing:**
   - Developed and tested a preprocessing pipeline for the Sleep-EDF dataset.
   - Key features of the script:
     - Matches PSG files with corresponding Hypnogram files using regex, handling cases where the last character in filenames changes.
     - Applies GPU-accelerated bandpass filtering (0.5–30 Hz) to EEG signals.
     - Adds sleep stage annotations from Hypnogram files to EEG data.
   - Processed files are saved in `.fif` format in the `processed_data/Sleep_edf` directory.

2. **Debugging and Enhancements:**
   - Fixed Hypnogram file matching using a regex pattern for variable last characters.
   - Logged detailed debugging information for file matching and saving processes.
   - Verified annotations were correctly added to EEG data.

3. **Results:**
   - Successfully processed several files, adding annotations with up to 123 events per file.
   - Verified output files include EEG signals and annotations:
     - Example: `sleep-cassette_SC4032E0_raw.fif`.

### Next Steps:
1. Resolve file-saving visibility issue:
   - Ensure `.fif` files are saved to the output directory and verify their integrity.
2. Extend preprocessing to cover all files in both **sleep-cassette** and **sleep-telemetry** datasets.
3. Begin feature extraction from processed EEG data for sleep stage classification.
4. Log progress in `LOG.md` and push updates to the repository.

## January 12, 2025

### Summary of Work:
1. **BCI Competition IV Preprocessing:**
   - Successfully preprocessed calibration and evaluation data from the BCI Competition IV Dataset 1 using the ASCII version of the dataset.
   - Handled large EEG data files (`_cnt.txt`) and their corresponding metadata (`_nfo.txt`) and markers (`_mrk.txt`).

2. **Issues Resolved:**
   - Mismatch between metadata channel count and EEG data columns:
     - Dynamically generated channel names (`Ch1`, `Ch2`, ..., `Ch59`) to match the data structure.
   - Improved data loading for large files using `np.genfromtxt` for efficiency.
   - Correctly added annotations from markers for calibration files.

3. **Output:**
   - Processed `.fif` files saved to the `processed_data` directory.
   - Example: 
     - `calib_BCICIV_calib_ds1a_1000Hz_raw.fif`
     - Successfully verified:
       - **59 EEG channels**
       - **200 annotations** (100 for `Class1`, 100 for `Class2`).

4. **Verification:**
   - Loaded and inspected `.fif` files using MNE.
   - Visualized raw EEG data and annotations using `raw.plot()`.

5. **Challenges Faced:**
   - Metadata inconsistencies in `_nfo.txt` files (e.g., incorrect channel count).
   - Long loading times for large `_cnt.txt` files, resolved with `np.genfromtxt`.

### Next Steps:
1. Process remaining calibration and evaluation files.
2. Verify all `.fif` files for completeness and correctness.
3. Begin feature extraction and classification for motor imagery tasks.
4. Log future progress and results.

## January 12, 2025

### UKB-EEG Preprocessing:
1. Successfully preprocessed EEG time series for all sets (F, N, O, S, Z).
2. Applied GPU-accelerated bandpass filtering (0.5–40 Hz).
3. Processed data saved in `.fif` format for each file:
   - Example: `F001_raw.fif` in `processed_data/F/`.

### Challenges Faced:
- No major challenges.

### Next Steps:
1. Inspect processed data for integrity and filtering.
2. Begin feature extraction or analysis on preprocessed EEG data.

## January 12, 2025

### UKB-EEG Preprocessing:
1. Successfully preprocessed EEG time series for all sets (F, N, O, S, Z).
2. Applied GPU-accelerated bandpass filtering (0.5–40 Hz).
3. Processed data saved in `.fif` format for each file:
   - Example: `F001_raw.fif` in `processed_data/F/`.

### Feature Extraction:
1. Extracted key statistical and EEG power band features from the preprocessed **UKB-EEG** dataset.
2. Features extracted include:
   - **Statistical Features**: Mean, Standard Deviation (STD), Skewness, Kurtosis.
   - **EEG Power Band Features**: Delta Power, Theta Power, Alpha Power, Beta Power, Gamma Power.
   - **Sample Entropy** was also computed for signal complexity analysis.
3. Stored extracted features in a structured format for model input.

### Model Training:
1. Chose **XGBoost Classifier** to model the EEG dataset for classification (F, N, O, S, Z).
2. Initial **cross-validation accuracy** of **86.8%** was achieved using the following features:
   - Alpha Power, Zero-Crossing Rate, Beta Power, Sample Entropy, Theta Power, Delta Power.
   - Hyperparameters used:
     - **n_estimators**: 100
     - **learning_rate**: 0.1
     - **max_depth**: 3
     - **min_child_weight**: 1
     - **subsample**: 1.0
     - **colsample_bytree**: 1.0

### Hyperparameter Tuning:
1. Conducted **grid search** for hyperparameter optimization.
2. Optimal hyperparameters found:
   - **n_estimators**: 200
   - **learning_rate**: 0.1
   - **max_depth**: 3
   - **min_child_weight**: 1
   - **subsample**: 1.0
   - **colsample_bytree**: 1.0
3. Best **mean cross-validation accuracy**: **0.8680**.

### Model Evaluation:
1. Applied **permutation importance** to assess feature importance.
2. Key features identified:
   - **Sample Entropy**, **Zero-Crossing Rate**, and **Beta Power** were the most important features for classification.
3. Visualized feature importance using **SHAP** values.
   - Addressed **dimension mismatch** issues by aggregating **SHAP values** across classes.

### Next Steps:
1. **Regularization**: Apply **L1 (Lasso)** or **L2 (Ridge)** regularization for further model tuning to reduce overfitting.
2. **Model Calibration**: Use **CalibratedClassifierCV** to refine probability predictions.
3. **Ensemble Methods**: Experiment with **Stacking** or **Voting Classifiers** to combine **XGBoost** with other models like **Random Forest** or **Logistic Regression** to improve model accuracy.
4. **Feature Engineering**: Explore additional feature engineering or data augmentation techniques to further improve the model.

### Challenges Faced:
1. Encountered **dimension mismatches** while visualizing **SHAP** values.
2. Warnings related to **GPU training** in **XGBoost**, but **GPU acceleration** was successfully enabled after modifying the configuration.

## January 13, 2025

### Regularization and Hyperparameter Tuning with XGBoost:

1. **Feature Selection**:
   - Selected top 6 features based on previous analysis:
     - `Alpha_Power`, `Zero-Crossing Rate`, `Beta_Power`, `Sample_Entropy`, `Theta_Power`, `Delta_Power`.
   
2. **Model Training**:
   - Used **XGBoost** for classification with GPU acceleration and cross-validation.
   - Regularization parameters (`reg_alpha`, `reg_lambda`) were adjusted to prevent overfitting.
   - Best mean cross-validation accuracy achieved: **0.8720** with the hyperparameters:
     - `n_estimators`: 200
     - `learning_rate`: 0.1
     - `max_depth`: 7
     - `min_child_weight`: 1
     - `subsample`: 0.8
     - `colsample_bytree`: 1.0
     - `reg_alpha`: 0.1
     - `reg_lambda`: 1
   
3. **Hyperparameter Tuning**:
   - Conducted manual hyperparameter tuning with a parameter grid.
   - Utilized GPU for faster training.
   - Best performing model parameters were recorded and verified with 5-fold cross-validation.
   - Achieved best mean CV accuracy of **0.8720** with regularization parameters included.

4. **Challenges Faced**:
   - Encountered some compatibility warnings related to XGBoost (`use_label_encoder` parameter).
   - There were a few issues with correct handling of class labels during cross-validation, which were resolved by properly encoding the target values.

5. **Next Steps**:
   - Explore more advanced techniques like **feature engineering**, **ensemble methods**, or **deep learning** to improve accuracy.
   - Fine-tune the model further and experiment with additional regularization and learning rate adjustments.
