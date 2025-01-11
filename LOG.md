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

