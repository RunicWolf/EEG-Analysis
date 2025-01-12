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

