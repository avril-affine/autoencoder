NUM_STEPS=25000
LR=0.0001
IMAGE_DIR='../GAN/flowers_resize'

python model.py \
    --num_steps=$NUM_STEPS \
    --learning_rate=$LR \
    --image_dir=$IMAGE_DIR
