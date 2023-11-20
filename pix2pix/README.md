Refer to https://github.com/saba99/pix2pix-Facades
Refer to https://github.com/phillipi/pix2pix

bash ./datasets/download.sh facades

python ./datasets/combine_A_and_B.py --fold_A ./datasets/facades/train/a --fold_B /datasets/facades/train/b --fold_AB /datasets/facades/train

python ./datasets/combine_A_and_B.py --fold_A ./datasets/facades/test/a --fold_B /datasets/facades/test/b --fold_AB /datasets/facades/test

python ./datasets/combine_A_and_B.py --fold_A ./datasets/facades/val/a --fold_B /datasets/facades/val/b --fold_AB /datasets/facades/val
