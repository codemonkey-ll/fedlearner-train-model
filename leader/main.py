# encoding=utf8
import logging

import tensorflow.compat.v1 as tf

import fedlearner.trainer as flt
import os

ROLE = 'leader'

parser = flt.trainer_worker.create_argument_parser()
parser.add_argument('--batch-size', type=int, default=32,
                    help='Training batch size.')
parser.add_argument('--clean-model', type=bool, default=True,
                    help='clean checkpoint and saved_model')
args = parser.parse_args()

args.save_checkpoint_steps = int(os.environ['CHECKPOINT_STEPS'])
args.checkpoint_path = os.environ["CHECKPOINT_PATH"]
args.export_path = os.environ["EXPORT_PATH"]


def apply_clean():
  if args.worker_rank == 0 and args.clean_model and tf.io.gfile.exists(args.checkpoint_path):
    tf.logging.info("--clean_model flag set. Removing existing checkpoint_path dir:"
                    " {}".format(args.checkpoint_path))
    tf.io.gfile.rmtree(args.checkpoint_path)

  if args.worker_rank == 0 and args.clean_model and args.export_path and tf.io.gfile.exists(args.export_path):
    tf.logging.info("--clean_model flag set. Removing existing savedmodel dir:"
                    " {}".format(args.export_path))
    tf.io.gfile.rmtree(args.export_path)

def input_fn(bridge, trainer_master=None):
  dataset = flt.data.DataBlockLoader(
      args.batch_size, ROLE, bridge, trainer_master).make_dataset()
  
  def parse_fn(example):
    feature_map = {}
    feature_map['example_id'] = tf.FixedLenFeature([], tf.string)
    feature_map['y'] = tf.FixedLenFeature([], tf.int64)
    features = tf.parse_example(example, features=feature_map)
    labels = {'y': features.pop('y')}
    return features, labels
  
  dataset = dataset.map(map_func=parse_fn,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(2)
  return dataset
  

def raw_serving_input_receiver_fn():
  features = {}
  features['embedding'] = tf.placeholder(dtype=tf.float32, shape=[1, 128], name='embedding')
  return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()

def model_fn(model, features, labels, mode):
  global_step = tf.train.get_or_create_global_step()
  xavier_initializer = tf.glorot_normal_initializer()

  fc1_size = 128
  with tf.variable_scope('leader'):
    w1f = tf.get_variable('w1f', shape=[
        fc1_size, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
    b1f = tf.get_variable(
        'b1f', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    embedding = model.recv('embedding', tf.float32, require_grad=True)
  else:
    embedding = features['embedding']
  
  logits = tf.nn.bias_add(tf.matmul(embedding, w1f), b1f)

  if mode == tf.estimator.ModeKeys.TRAIN:
    y = tf.dtypes.cast(labels['y'], tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.math.reduce_mean(loss)

    # cala auc
    pred = tf.math.sigmoid(logits)
    _, auc = tf.metrics.auc(labels=y, predictions=pred)

    logging_hook = tf.train.LoggingTensorHook(
        {"loss": loss, "auc": auc}, every_n_iter=10)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = model.minimize(optimizer, loss, global_step=global_step)
    return model.make_spec(mode, loss=loss, train_op=train_op,
                            training_hooks=[logging_hook])

  if mode == tf.estimator.ModeKeys.PREDICT:
    return model.make_spec(mode, predictions=logits)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s'
    )
    apply_clean()
    flt.trainer_worker.train(
        ROLE, args, input_fn,
        model_fn, raw_serving_input_receiver_fn)
