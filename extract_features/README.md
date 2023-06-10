# TSP Feature Extraction

Follow the data preprocessing instructions described [here](../data) before extracting features. We provide scripts for feature extraction using the released pretrained models or using a local checkpoint.

### From a Local Checkpoint
Use the `extract_features_from_a_local_checkpoint.sh` script to extract features from a local checkpoint. You need to manually set the same variables above plus the following 2 variables instead of `RELEASED_CHECKPOINT`:
- `LOCAL_CHECKPOINT`: Path to the local checkpoint `.pth` file.
- `BACKBONE`: The backbone used in the local checkpoint: `r2plus1d_34`, `i3d`, or `x3d`.

## Post Processing Output
The feature extraction script will output a `.pkl` file for each video. Merge all the `.pkl` files into one `.h5` file as follows:

```
python merge_pkl_files_into_one_h5_feature_file.py --features-folder <path/to/feature/output/folder/> --output-h5 <features_filenames.h5>
```
