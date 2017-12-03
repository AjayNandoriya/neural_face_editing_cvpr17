modify prepare_script.m by specifying image directory to be used to generate training data.
Run prepare_script.m using Matlab
Use neural_face_relighting.py to train the network
Use neural_face_relighting_2input.py to relight

The train_file format is as follow.The values are separated by space.
<input image> <output/reconstruct image> <mask image (1-channel)> <normal image> <27 float values for Light>


#sample command to train
CUDA_VISIBLE_DEVICES=1 python neural_face_relighting.py --mode train --train_file ../tmp/datafile1.txt --val_file ../tmp/datafile1.txt --output_dir models/cvpr17_check --input_dir ../tmp/ --max_epochs 1000 --checkpoint models/cvpr17_check/ --normal_weight 100 --mask_weight 100 --lcoeff_weight 100