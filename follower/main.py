import logging
import datetime

import tensorflow.compat.v1 as tf 
import fedlearner.trainer as flt 
import os

from slot_2_bucket import slot_2_bucket

_SLOT_2_IDX = {pair[0]: i for i, pair in enumerate(slot_2_bucket)}
_SLOT_2_BUCKET = slot_2_bucket
ROLE = "follower"

parser = flt.trainer_worker.create_argument_parser()
parser.add_argument('--batch-size', type=int, default=32,
                    help='Training batch size.')
parser.add_argument('--clean-model', type=bool, default=True,
                    help='clean checkpoint and saved_model')
args = parser.parse_args()

args.save_checkpoint_steps = int(os.environ['CHECKPOINT_STEPS'])
args.checkpoint_path = os.environ["CHECKPOINT_PATH"]
args.export_path = os.environ["EXPORT_PATH"]
args.sparse_estimator = bool(os.environ['SPARSE_ESTIMATOR'])

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
    feature_map["example_id"] = tf.FixedLenFeature([], tf.string)
    feature_map['fids'] = tf.VarLenFeature(tf.int64)
    # feature_map['y'] = tf.FixedLenFeature([], tf.int64)
    features = tf.parse_example(example, features=feature_map)
    # labels = {'y': features.pop('y')}
    labels = {'y': tf.constant(0)}
    return features, labels
  dataset = dataset.map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(2)
  return dataset
  
  # feature_map = {"fids": tf.VarLenFeature(tf.int64)}
  # feature_map['example_id'] = tf.FixedLenFeature([], tf.string)
  # record_batch = dataset.make_batch_iterator().get_next()
  # features = tf.parse_example(record_batch, features=feature_map)
  # return features, None

def raw_serving_input_receiver_fn():
  features = {}
  features['fids_indices'] = tf.placeholder(dtype=tf.int64, shape=[None], name='fids_indices')
  features['fids_values'] = tf.placeholder(dtype=tf.int64, shape=[None], name='fids_values')
  features['fids_dense_shape'] = tf.placeholder(dtype=tf.int64, shape=[None], name='fids_dense_shape')
  return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()


def model_fn(model, features, labels, mode):

  def sum_pooling(embeddings, slots):
    slot_embeddings = []
    for slot in slots:
      slot_embeddings.append(embeddings[_SLOT_2_IDX[slot]])
    if len(slot_embeddings) == 1:
      return slot_embeddings[0]
    return tf.add_n(slot_embeddings)

  global_step = tf.train.get_or_create_global_step()
  num_slot, embed_size = len(_SLOT_2_BUCKET), 8
  xavier_initializer = tf.glorot_normal_initializer()

  flt.feature.FeatureSlot.set_default_bias_initializer(
        tf.zeros_initializer())
  flt.feature.FeatureSlot.set_default_vec_initializer(
        tf.random_uniform_initializer(-0.0078125, 0.0078125))
  flt.feature.FeatureSlot.set_default_bias_optimizer(
        tf.train.FtrlOptimizer(learning_rate=0.01))
  flt.feature.FeatureSlot.set_default_vec_optimizer(
        tf.train.AdagradOptimizer(learning_rate=0.01))

  # deal with input cols
  categorical_embed = []
  num_slot, embed_dim = len(_SLOT_2_BUCKET), 8

  with tf.variable_scope("follower"):
    for slot, bucket_size in _SLOT_2_BUCKET:
      fs = model.add_feature_slot(slot, bucket_size)
      fc = model.add_feature_column(fs)
      categorical_embed.append(fc.add_vector(embed_dim))


  # concate all embeddings
  slot_embeddings = categorical_embed
  concat_embedding = tf.concat(slot_embeddings, axis=1)
  output_size = len(slot_embeddings) * embed_dim

  model.freeze_slots(features)

  with tf.variable_scope("follower"):
    fc1_size, fc2_size, fc3_size = 512, 256, 128
    w1 = tf.get_variable('w1', shape=[output_size, fc1_size], dtype=tf.float32,
                        initializer=xavier_initializer)
    b1 = tf.get_variable(
        'b1', shape=[fc1_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    w2 = tf.get_variable('w2', shape=[fc1_size, fc2_size], dtype=tf.float32,
                        initializer=xavier_initializer)
    b2 = tf.get_variable(
        'b2', shape=[fc2_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    w3 = tf.get_variable('w3', shape=[fc2_size, fc3_size], dtype=tf.float32,
                        initializer=xavier_initializer)
    b3 = tf.get_variable(
        'b3', shape=[fc3_size], dtype=tf.float32, initializer=tf.zeros_initializer())

  act1_l = tf.nn.relu(tf.nn.bias_add(tf.matmul(concat_embedding, w1), b1))
  act1_l = tf.layers.batch_normalization(act1_l, training=True)
  act2_l = tf.nn.relu(tf.nn.bias_add(tf.matmul(act1_l, w2), b2))
  act2_l = tf.layers.batch_normalization(act2_l, training=True)
  embedding = tf.nn.relu(tf.nn.bias_add(tf.matmul(act2_l, w3), b3))
  embedding = tf.layers.batch_normalization(embedding, training=True)

  if mode == tf.estimator.ModeKeys.TRAIN:
    embedding_grad = model.send('embedding', embedding, require_grad=True)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = model.minimize(
        optimizer, embedding, grad_loss=embedding_grad, global_step=global_step)
    return model.make_spec(mode, loss=tf.math.reduce_mean(embedding), train_op=train_op)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    return model.make_spec(mode, predictions={'embedding': embedding})

if __name__ == '__main__':
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s'
  )
  apply_clean()
  flt.trainer_worker.train(
      ROLE, args, input_fn,
      model_fn, raw_serving_input_receiver_fn)