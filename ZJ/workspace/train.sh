MODELS_DIR="./models"
if [ ! -d $MODELS_DIR ]; then
    mkdir models
fi
if [ ! "$(ls -A $DIR)" ]; then
     git clone https://github.com/tensorflow/models/
fi

FILE=./tf-data/trainall-0-of-2.tfrecord
if [ ! -f $FILE ]; then
	echo "Converting data ..."
	mkdir tf-data
	python3 convert_data.py
fi

FILE=./models/inception_resnet_v2_2016_08_30
if [ ! -f $FILE ]; then
	wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
	mkdir models
	tar -xvf inception_resnet_v2_2016_08_30.tar.gz
	mv inception_resnet_v2_2016_08_30.ckpt ./models/
	rm inception_resnet_v2_2016_08_30.tar.gz
fi

python3 train_and_test.py \
--stage=trainall \
--network_name=inception_resnet_v2 \
--num_epochs=50
