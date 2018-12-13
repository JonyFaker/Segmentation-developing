for file in ./test_img/val/*
do 
  echo $file

  #TEST_IMG=ADE_val_00001519.jpg
  TEST_IMG=$file
  MODEL_PATH=./ckpt/baseline-resnet50_dilated8-ppm_bilinear_deepsup-ngpus2-batchSize4-imgMaxSize700-paddingConst8-segmDownsampleRate8-LR_encoder0.02-LR_decoder0.02-epoch20-decay0.0001-fixBN0
  RESULT_PATH=./result_sun

  # ENCODER=$MODEL_PATH/encoder_epoch_20.pth
  # DECODER=$MODEL_PATH/decoder_epoch_20.pth

  # if [ ! -e $ENCODER ]; then
  #   mkdir $MODEL_PATH
  # fi
  # if [ ! -e $ENCODER ]; then
  #   wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
  # fi
  # if [ ! -e $DECODER ]; then
  #   wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
  # fi
  # if [ ! -e $TEST_IMG ]; then
  #   wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu//data/ADEChallengeData2016/images/validation/$TEST_IMG
  # fi

  python3 -u test.py \
    --model_path $MODEL_PATH \
    --test_img $TEST_IMG \
    --num_class 14 \
    --arch_encoder resnet50_dilated8 \
    --arch_decoder ppm_bilinear_deepsup \
    --suffix _epoch_11.pth \
    --fc_dim 2048 \
    --result $RESULT_PATH \
    --gpu_id 1

done
