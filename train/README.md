# TSP Training

We provide four training scripts:
- `train_tsp_on_thumos14.sh`: pretraining R(2+1)D-34 encoder with TSP on THUMOS14
- `train_tac_on_thumos14.sh`: pretraining R(2+1)D-34 encoder with TAC on THUMOS14 

## Launching the Training Scripts

Before launching each script, you need to manually set **the following variable** inside each file:
- `ROOT_DIR`: The root directory of either the ActivityNet or THUMOS14 videos. Follow the data preprocessing instructions and subfolders naming described [here](../data).

## Experiment Output

- Checkpoint per epoch (*e.g.,* `epoch_3.pth`): a `.pth` file containing the state dictionary of the model, optimizer, and learning rate scheduler. The checkpoint files can be used to resume the training (use `--resume` and `--start-epoch` input parameters in `train.py`) or to extract features (use the scripts [here](../extract_features)).
- Metric results file (`results.txt`): A log of the metrics results on the validation subset after each epoch. We choose the best pretrained model based on the epoch with the highest `Avg Accuracy` value.
