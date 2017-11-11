FILE=./tf-data/test-0-of-2.tfrecord
if [ -f $FILE ]; then
	echo "FILE $FILE exists."
else
	echo "Converting data ..."
	mkdir tf-data
	python3 convert_data.py
fi

python3 train_and_test.py \
--stage=test

