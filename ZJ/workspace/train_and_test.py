import sys
sys.path.insert(0, "./tensorflow-slim/research/slim")
import os
from tensorflow.python.platform import tf_logging as logging
import time
import tensorflow as tf
from preprocessing import inception_preprocessing
from nets import inception
from nets.nasnet import nasnet
import tensorflow.contrib.slim as slim
import csv

tf.app.flags.DEFINE_string(
    'stage', '', 'The function to run, one of [train, val, test, val-train, trainall].'
                 'val-train will generate validation score with training data'
                 'trainall will train the model with all available data'
)

tf.app.flags.DEFINE_string(
    'network_name', 'inception_resnet_v2',
    'The network model to use, on of [inception_resnet_v2, inception_v4, nasnet_large]'
)

tf.app.flags.DEFINE_integer(
    'num_epochs', 200,
    'Number of epochs to train.'
    'You may also run \'touch .stop\' to stop the training after current epoch.')

FLAGS = tf.app.flags.FLAGS

stage = FLAGS.stage
network_name = FLAGS.network_name
num_epochs = FLAGS.num_epochs

data_file = stage
is_train = stage in ['train', 'trainall']
if stage == 'val-train':
    data_file = 'train'
    stage = 'val'
elif stage == 'trainall':
    stage = 'train'

reg = 3.0

proj_dir = '.'
data_dir = os.path.join(proj_dir, 'tf-data')
log_dir = os.path.join(proj_dir, 'log-' + network_name)
stop_file = os.path.join(proj_dir, '.stop')

initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

num_classes = 132


class NetworkInfo:
    def __init__(self, network_fn, arg_scopes, preprocess_fn, checkpoint, exclude, batch_size):
        self.network_fn = network_fn
        self.arg_scopes = arg_scopes
        self.preprocessing_fn = preprocess_fn
        self.checkpoint = checkpoint
        self.exclude = exclude
        self.batch_size = batch_size

    def get_network(self):
        return self.network_fn


network_map = {
    'inception_v4': NetworkInfo(
        network_fn=inception.inception_v4,
        arg_scopes=inception.inception_v4_arg_scope,
        preprocess_fn=inception_preprocessing,
        checkpoint=os.path.join(proj_dir, 'models/inception_v4.ckpt'),
        exclude=['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
        batch_size=48),
    'inception_resnet_v2': NetworkInfo(
        network_fn=inception.inception_resnet_v2,
        arg_scopes=inception.inception_resnet_v2_arg_scope,
        preprocess_fn=inception_preprocessing,
        checkpoint=os.path.join(proj_dir, 'models/inception_resnet_v2_2016_08_30.ckpt'),
        exclude=['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'],
        batch_size=32),
    'nasnet_large': NetworkInfo(
        network_fn=nasnet.build_nasnet_large,
        arg_scopes=nasnet.nasnet_large_arg_scope,
        preprocess_fn=inception_preprocessing,
        checkpoint=os.path.join(proj_dir, 'models/nasnet-a_large_04_10_2017'),
        exclude=['final_layer/FC', 'aux_11/aux_logits/FC', 'cell_stem_0/comb_iter_0/left/global_step'],
        batch_size=8),
}
network = network_map[network_name]
image_size = network.network_fn.default_image_size

items_to_descriptions = {
    'image': 'A 3-channel RGB image of dishes',
    'label': 'A integer label denote the category of the dish'
}

labels_to_name_dict = {}
for i in range(num_classes):
    labels_to_name_dict[i] = str(i)


def get_dataset(name):
    files = [os.path.join(data_dir, file)
             for file in os.listdir(data_dir)
             if file.startswith(name + '-')]
    print(files)
    num_samples = 0
    for fn in files:
        for record in tf.python_io.tf_record_iterator(fn):
            num_samples += 1

    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/name': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'name': slim.tfexample_decoder.Tensor('image/name')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(
        data_sources=os.path.join(data_dir, name + '-*.tfrecord'),
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=num_classes,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)
    return dataset


def load_batch(dataset, is_training=True):
    # with tf.device('/cpu:0'):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=is_train,
        common_queue_capacity=24 + 3 * network.batch_size,
        common_queue_min=24)

    raw_image, label, name = data_provider.get(['image', 'label', 'name'])

    image = network.preprocessing_fn.preprocess_image(raw_image,
                                                      image_size, image_size, is_training)

    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [image_size, image_size])
    raw_image = tf.squeeze(raw_image)

    images, raw_images, labels, name = tf.train.batch(
        [image, raw_image, label, name],
        batch_size=network.batch_size,
        num_threads=4,
        capacity=4 * network.batch_size,
        allow_smaller_final_batch=True)

    return images, raw_images, labels, name


def train():
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        dataset = get_dataset(data_file)

        num_batches_per_epoch = int(dataset.num_samples / network.batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        images, _, labels, name = load_batch(dataset, is_training=is_train)
        with slim.arg_scope(network.arg_scopes()):
            logits, end_points = network.network_fn(images,
                                                    num_classes=dataset.num_classes,
                                                    dropout_keep_prob=0.5,
                                                    is_training=is_train)
        predictions = tf.argmax(end_points['Predictions'], 1)

        variables_to_restore = slim.get_variables_to_restore(exclude=network.exclude)
        # print('exclude', network.exclude)
        # print('to restore')
        # for v in variables_to_restore:
        #     print(v)
        # return
        # print('trainable', slim.get_trainable_variables())
        # print('', slim.get_variables('aux_logits'))
        # print('', slim.get_variables('aux_11'))
        # print('', slim.get_variables('aux_'))

        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        # reg_loss = tf.losses.get_regularization_loss()
        # normal_loss = tf.losses.get_total_loss(add_regularization_losses=False)
        # total_loss = normal_loss + reg_loss * reg
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        # print('loss', tf.losses.get_losses())
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total_reg_loss = sum(reg_losses)
        # total_reg_loss_2 = tf.losses.get_regularization_loss()
        # total_loss_2 = tf.losses.get_total_loss(add_regularization_losses=False)
        # for v in reg_losses:
        #     print(v)

        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        my_summary = list()
        my_summary.append(tf.summary.scalar('losses/Total_Loss', total_loss))
        # my_summary.append(tf.summary.scalar('losses/reg_loss', total_reg_loss))
        my_summary.append(tf.summary.scalar('accuracy', accuracy))
        my_summary.append(tf.summary.scalar('learning_rate', lr))
        my_summary_op = tf.summary.merge(my_summary)

        # print(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # return

        def train_step(sess, train_op, global_step):
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
            return total_loss, global_step_count

        saver = tf.train.Saver(variables_to_restore)

        if tf.gfile.IsDirectory(network.checkpoint):
            checkpoint_file = tf.train.latest_checkpoint(network.checkpoint)
        else:
            checkpoint_file = network.checkpoint

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)

        with sv.managed_session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    if os.path.exists(stop_file):
                        os.remove(stop_file)
                        break

                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


def eval():
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        dataset = get_dataset(data_file)

        num_batches_per_epoch = int(dataset.num_samples / network.batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        images, _, labels, name = load_batch(dataset, is_training=is_train)
        with slim.arg_scope(network.arg_scopes()):
            logits, end_points = network.network_fn(images, num_classes=dataset.num_classes, is_training=is_train)
        predictions = tf.argmax(end_points['Predictions'], 1)

        num_match = tf.count_nonzero(tf.equal(predictions, labels))

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        if tf.gfile.IsDirectory(network.checkpoint):
            checkpoint_file = tf.train.latest_checkpoint(network.checkpoint)
        else:
            checkpoint_file = network.checkpoint

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)

        if stage == 'val':
            start = time.time()
            with sv.managed_session() as sess:
                score = 0
                for i in range(num_batches_per_epoch):
                    curr = sess.run(num_match)
                    score += curr
                    print(curr, score / ((i + 1) * network.batch_size))
                print(score, '/', num_batches_per_epoch * network.batch_size,
                      score / (num_batches_per_epoch * network.batch_size))
            end = time.time()
            print(end - start, 'seconds')
        else:
            start = time.time()
            res = []
            done_set = set()
            with sv.managed_session() as sess:
                while len(done_set) < dataset.num_samples:
                    name_v, pred_v = sess.run([name, predictions])
                    # print('name', name_v, 'pred', pred_v)
                    for k in range(name_v.shape[0]):
                        res.append([name_v[k].decode('ascii'), str(pred_v[k])])
                        done_set.add(name_v[k].decode('ascii'))
                        # res.append([name_v[k].decode('ascii'), str(pred_v[k])])
                    print(name_v.shape[0], len(done_set), len(res), dataset.num_samples)
            end = time.time()
            print(end - start, 'seconds')
            #print(res)
            res.sort(key=lambda f: int(''.join(filter(str.isdigit, f[0]))))
            #print(res)

            with open(os.path.join(proj_dir, 'test.csv'), mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(['image_name', 'category'])
                last_row = ['', '']
                dup = 0
                for r in res:
                    if r[0] == last_row[0]:
                        dup += 1
                        if last_row[1] != r[1]:
                            print(last_row, r)
                    else:
                        writer.writerow(r)
                    last_row = r
            print('#res', len(res) - dup, '#dup', dup)
            print('#images', dataset.num_samples)


if is_train:
    train()
else:
    eval()
