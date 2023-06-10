import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-path', required=True,
                        help='Path to the directory containing the videos files')
    parser.add_argument('--metadata-csv-filename', required=True,
                        help='Path to the metadata CSV file')

    parser.add_argument('--backbone', default='r2plus1d_34',
                        choices=['r2plus1d_34', 'i3d', 'x3d'],
                        help='Encoder backbone architecture. '
                             'Supported backbones are r2plus1d_34, i3d and x3d')
    parser.add_argument('--device', default='cuda',
                        help='Device to train on')

    parser.add_argument('--local-checkpoint', default=None,
                        help='Path to checkpoint on disk. If set, then read checkpoint from local disk.')

    parser.add_argument('--clip-length', default=16, type=int,
                        help='Number of frames per clip')
    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled')
    parser.add_argument('--stride', default=1, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips')

    parser.add_argument('--batch-size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--workers', default=6, type=int,
                        help='Number of data loading workers')

    parser.add_argument('--output-dir', required=True,
                        help='Path for saving features')

    args = parser.parse_args()

    return args
