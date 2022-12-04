"""CIFAR dataset input module.
"""
import pandas as pd
import tensorflow as tf
import os, sys
import tarfile
from six.moves import urllib
from PIL import Image
import numpy as np
from numpy import asarray

def build_input(dataset,data_path, batch_size, standardize_images, mode):
  """Build MNIST dataset.

  Args:
    data_path: Filename for data.
    batch_size: Input batch size.
    mode: Either 'train' or 'eval'.
  Returns:
    images: Batches of images. [batch_size, image_size, image_size, 3]
    labels: Batches of labels. [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  wiki_list = pd.read_csv(str(data_path)+'/'+str(dataset)+"/wiki_list.txt", sep="\t", header=None,names=['file_path','age','gender'])
  train_test_ratio = 0.8
  train_size = int(len(wiki_list) * train_test_ratio)
  wiki_list["age"] = wiki_list["age"].astype(int)
  train_image_directory = wiki_list["file_path"][:train_size].apply(lambda x: str(data_path)+'/'+str(dataset)+"/"+str(x))
  train_labels_list = wiki_list["age"][:train_size]
  test_image_directory = wiki_list["file_path"][train_size:].apply(lambda x: str(data_path)+'/'+str(dataset)+"/"+str(x))
  test_labels_list = wiki_list["age"][train_size:]
  if mode == 'train':
    def image_generator():
      for image_file, label in zip(train_image_directory, train_labels_list.to_list()):
        image = Image.open(os.path.join(image_file)).convert('RGB')
        image = image.resize((32, 32))
        image = np.array(image) / 255.0 - 0.5
        if np.isnan(image).any():
          continue
        yield image, np.array([label])
    dataset = tf.data.Dataset.from_generator(image_generator, output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([32, 32, 3]), tf.TensorShape([1])))

    dataset  = dataset.shuffle(10000).repeat()
  elif mode == 'eval':
    dataset = test(test_image_directory, test_labels_list,  standardize_images)
  dataset  = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()

def test(image_directory,labels_list, standardize_images):
  """tf.data.Dataset object for MNIST test data."""

  def image_generator():
    for image_file, label in zip(image_directory, labels_list.to_list()):
      image = Image.open(os.path.join(image_file)).convert('RGB')
      image = image.resize((32, 32))
      image = np.array(image) / 255.0 - 0.5
      yield image, np.array([label])

  return tf.data.Dataset.from_generator(image_generator, output_types=(tf.float32, tf.float32),
                                        output_shapes=((32, 32, 3), (1)))


