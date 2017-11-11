import math
import csv
import os
import sys
import numpy as np
import tensorflow as tf

in_folder = './data'
out_folder = './tf-data'
train_x_folder = os.path.join(in_folder, 'train_images')
train_y_file = os.path.join(in_folder, 'train.csv')
test_x_folder = os.path.join(in_folder, 'test_images')

validation_size = 0.3
num_shards = 2


def load_labels(fn):
    l = list()
    with open(fn) as f:
        reader = csv.reader(f)
        for row in reader:
            l.append(row)
    arr = np.array(l[1:])
    return arr


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id, fn):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/name': bytes_feature(fn),
    }))


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _convert_dataset(name, dir, samples):
    num_per_shard = int(math.ceil(len(samples) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = os.path.join(out_folder, '%s-%d-of-%d.tfrecord' %
                                               (name, shard_id, num_shards))

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(samples))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(samples), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        fn = os.path.join(dir, samples[i, 0])
                        id = int(samples[i, 1])
                        image_data = tf.gfile.FastGFile(fn, 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        example = image_to_tfexample(
                            image_data, b'jpg', height, width, id, bytes(samples[i, 0], 'ascii'))
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


if dir().count('train_x_folder') and dir().count('train_y_file'):
    print('train')
    # exit()
    all_samples = load_labels(train_y_file)
    np.random.shuffle(all_samples)
    num_validation = int(validation_size * all_samples.shape[0])
    val_samples = all_samples[:num_validation]
    train_samples = all_samples[num_validation:]
    print('#images:', all_samples.shape[0], '#val:', num_validation,
          '#train', train_samples.shape[0])
    # _convert_dataset("train", train_x_folder, train_samples)
    # _convert_dataset("val", train_x_folder, val_samples)

    _convert_dataset('trainall', train_x_folder, all_samples)


if dir().count('test_x_folder'):
    print('test')
    files = os.listdir(test_x_folder)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print('#files:', len(files))
    samples = np.c_[np.array(files), np.empty(len(files))]
    samples[:, 1] = '0'
    #b = bytes(samples[0,0], 'ascii')
    #print(b)
    _convert_dataset('test', test_x_folder, samples)
