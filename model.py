import tensorflow as tf
import numpy as np
import os
import re

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


# Optimization parameters.
tf.app.flags.DEFINE_integer('num_steps', 2000,
                            """How many training steps to run.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0002,
                          """Learning rate suggested in paper.""")
tf.app.flags.DEFINE_float('weight_init', 0.02,
                          """Weight initialization standard deviation.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size for training.""")
tf.app.flags.DEFINE_integer('in_channels', 3,
                            """Number of input channels.""")
tf.app.flags.DEFINE_integer('img_size', 64,
                            """Dimension of image assumed to be square""")

tf.app.flags.DEFINE_string('summary_dir', 'logs/',
                           """Path of where to store the summary files.""")
tf.app.flags.DEFINE_string('image_dir', 'flickr_resize/',
                           """Path of where to store the summary files.""")


INPUT_CHANNELS = FLAGS.in_channels
IMAGE_SIZE = FLAGS.img_size

def get_random_input_images(sess, image_dir, batch_size,
                            image_data_tensor, decode_tensor):
    filenames = [f for f in os.listdir(image_dir) if not f.startswith('.')]
    images = []
    for _ in xrange(batch_size):
        index = np.random.randint(0, len(filenames))
        image_path = os.path.join(image_dir, filenames[index])
        image_data = gfile.FastGFile(image_path, 'rb').read()
        image = sess.run(decode_tensor,
                         feed_dict={image_data_tensor: image_data})
        image = image * 2. / 255. - 1.
        images.append(image.flatten())
    return images


def fc_layer(in_tensor, in_size, out_size, activation_func, weight_init, name):
    initializer = tf.random_normal_initializer(stddev=weight_init)
    weights = tf.get_variable(name + '/weights',
                              shape=[in_size, out_size],
                              initializer=initializer)
    bias = tf.get_variable(name + '/bias',
                           shape=[out_size],
                           initializer=tf.constant_initializer())
    affine = tf.nn.bias_add(tf.matmul(in_tensor, weights, name=name + '/mul'),
                            bias, name=name + '/affine')
    out_tensor = activation_func(affine, name=name + '/activations')
    tf.histogram_summary('summary/weights/' + name, weights)
    tf.histogram_summary('summary/activations/' + name, out_tensor)
    return out_tensor


def model(input_tensor):
    dim = IMAGE_SIZE * IMAGE_SIZE * INPUT_CHANNELS
    flat_tensor = tf.reshape(input_tensor, [-1, dim])
    fc1 = fc_layer(flat_tensor, dim, dim / 2, tf.nn.relu, 0.02, 'fc1')
    fc2 = fc_layer(fc1, dim / 2, dim / 4, tf.nn.relu, 0.02, 'fc2')
    features = fc_layer(fc2, dim / 4, 1024, tf.nn.relu, 0.02, 'features')
    fc4 = fc_layer(features, 1024, dim / 4, tf.nn.relu, 0.02, 'fc4')
    fc5 = fc_layer(fc4, dim / 4, dim / 2, tf.nn.relu, 0.02, 'fc5')
    out_tensor = fc_layer(fc5, dim / 2, dim, tf.nn.relu, 0.02, 'output')
    return features, out_tensor


def add_optimization(in_tensor, out_tensor, learning_rate):
    loss = tf.nn.l2_loss(in_tensor - out_tensor, 'loss')
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return loss, opt


def main(_):
    flat_dim = IMAGE_SIZE * IMAGE_SIZE * INPUT_CHANNELS

    in_tensor = tf.placeholder(tf.float32,
                               shape=[None, flat_dim],
                               name='input_image')
    features, out_tensor = model(in_tensor)

    # Read image tensors.
    image_data_tensor = tf.placeholder(tf.string)
    decode_tensor = tf.image.decode_jpeg(image_data_tensor,
                                         channels=INPUT_CHANNELS)

    # Add update steps
    loss_tensor, train_opt = add_optimization(in_tensor, out_tensor,
                                              FLAGS.learning_rate)

    # Create graph
    sess = tf.Session()
    saver = tf.train.Saver()

    checkpoint_file = os.path.join(FLAGS.summary_dir, 'checkpoint')
    step_0 = 0
    if os.path.exists(checkpoint_file):
        print 'Restoring checkpoint'
        with open(checkpoint_file, 'r') as f:
            line = f.readline().strip()
        model_ckpt = line.split(': ')[1]
        model_ckpt = model_ckpt.strip('"')
        step_0 = int(model_ckpt.split('-')[-1])
        model_ckpt = os.path.join(FLAGS.summary_dir, model_ckpt)
        saver.restore(sess, model_ckpt)
    else:
        init = tf.initialize_all_variables()
        sess.run(init)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)
    out_image = tf.cast((out_tensor + 1) * 255 / 2,
                         tf.uint8)
    out_image = tf.reshape(out_image,
                           [-1,
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            INPUT_CHANNELS],
                           name='output_image')

    img_summary = tf.image_summary('Output Images', out_image, 10)

    n_train = len([f for f in os.listdir(FLAGS.image_dir)
                   if not f.startswith('.')])
    n_epoch = n_train / FLAGS.batch_size

    for step in xrange(step_0 + 1, FLAGS.num_steps):
        batch_imgs = get_random_input_images(sess,
                                             FLAGS.image_dir,
                                             FLAGS.batch_size,
                                             image_data_tensor,
                                             decode_tensor)
        loss, merged_str, _ = sess.run([loss_tensor, merged, train_opt],
                                       feed_dict={in_tensor: batch_imgs})
        print '{} | Step {} | Loss = {}'.format(datetime.now(),
                                                step,
                                                loss)

        writer.add_summary(merged_str, step)

        if step % 100 == 0 or step + 1 == FLAGS.num_steps:
            print 'Writing test image'
            img_str = sess.run(img_summary,
                               feed_dict={in_tensor: batch_imgs})
            writer.add_summary(img_str, step)
            model_file = os.path.join(FLAGS.summary_dir, 'model.ckpt')
            saver.save(sess, model_file, global_step=step)

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ['output'])
    with gfile.FastGFile(os.path.join(FLAGS.summary_dir, 'ae_graph.pb'), 
                         'wb') as f:
        f.write(output_graph_def.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
