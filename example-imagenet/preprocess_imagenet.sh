# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Preprocess imagenet raw data to create TFRecords
# Need to specify your imagenet username and access key
python3 preprocess_imagenet.py --raw_data_dir=/fsx/imagenet2012/Data/CLS-LOC --local_scratch_dir=/fsx/original

# Resize training & validation data, maintaining the aspect ratio, 
python3 tensorflow_image_resizer.py -d imagenet -i /fsx/original/train  -o  /fsx/resized/train \
	--subset_name train --num_preprocess_threads 60 --num_intra_threads 2 --num_inter_threads 2
python3 tensorflow_image_resizer.py -d imagenet -i /fsx/original/val  -o  /fsx/resized/val \
	--subset_name validation --num_preprocess_threads 60 --num_intra_threads 2 --num_inter_threads 2
