import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training script')

    parser.add_argument('--root-dir', required=True,
                        help='Path to root directory containing the videos files')
    parser.add_argument('--train-subdir', default='train',
                        help='Training subdirectory inside the root directory')
    parser.add_argument('--valid-subdir', default='valid',
                        help='Validation subdirectory inside the root directory')
    parser.add_argument('--train-csv-filename', required=True,
                        help='Path to the training CSV file')
    parser.add_argument('--valid-csv-filename', required=True,
                        help='Path to the validation CSV file')
    parser.add_argument('--label-columns', nargs='+', required=True,
                        help='Names of the label columns in the CSV files')
    parser.add_argument('--label-mapping-jsons', nargs='+', required=True,
                        help='Path to the mapping of each label column')
    parser.add_argument('--loss-alphas', nargs='+', default=[1.0, 1.0], type=float,
                        help='A list of the scalar alpha with which to weight each label loss')
    parser.add_argument('--global-video-features',
                        help='Path to the h5 file containing global video features (GVF). '
                             'If not given, then train without GVF.')

    parser.add_argument('--backbone', default='r2plus1d_34',
                        choices=['r2plus1d_34', 'i3d', 'x3d'],
                        help='Encoder backbone architecture. '
                             'Supported backbones are r2plus1d_34, i3d and x3d')
    parser.add_argument('--device', default='cuda',
                        help='Device to train on')

    parser.add_argument('--clip-length', default=16, type=int,
                        help='Number of frames per clip')
    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled')
    parser.add_argument('--clips-per-segment', default=5, type=int,
                        help='Number of clips sampled per video segment')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loading workers')

    parser.add_argument('--epochs', default=8, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--backbone-lr', default=0.0001, type=float,
                        help='Backbone layers learning rate')
    parser.add_argument('--fc-lr', default=0.002, type=float,
                        help='Fully-connected classifiers learning rate')
    parser.add_argument('--lr-warmup-epochs', default=2, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--lr-milestones', nargs='+', default=[4, 6], type=int,
                        help='Decrease lr on milestone epoch')
    parser.add_argument('--lr-gamma', default=0.01, type=float,
                        help='Decrease lr by a factor of lr-gamma at each milestone epoch')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum')
    parser.add_argument('--weight-decay', default=0.005, type=float,
                        help='Weight decay')

    parser.add_argument('--print-freq', default=100, type=int,
                        help='Print frequency in number of batches')
    parser.add_argument('--output-dir', required=True,
                        help='Path for saving checkpoints and results output')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='Start epoch')

    args = parser.parse_args()

    assert len(args.label_columns) == len(args.label_mapping_jsons) and len(args.label_columns) == len(
        args.loss_alphas), \
        (f'The parameters label-columns, label-mapping-jsons, and loss-alphas must have the same length. '
         f'Got len(label-columns)={len(args.label_columns)}, len(label-mapping-jsons)={len(args.label_mapping_jsons)}, '
         f'and len(loss-alphas)={len(args.loss_alphas)}')

    return args
