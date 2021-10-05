import glob
import os.path
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir',
                           './dataset/For_Training_320_240/',
                           'Dataset directory')

tf.app.flags.DEFINE_string('input_image_format', 'png', 'Input image format.')


def _separate_sets(filenames, validation_percent=0.2):
  """Separates train and test set randomly and returns them.

  Args:
    filenames: All filenames in inputset.
    validation_percent: Validation set percentage.
  """
  filenames_rand = filenames
  np.random.shuffle(filenames_rand)
  num_total_files = len(filenames_rand)
  num_validation_files = int(num_total_files * validation_percent) or 1
  validation_set = filenames_rand[:num_validation_files]
  train_set = filenames_rand[num_validation_files:]

  return train_set, validation_set


def _write_to_text_file(save_dir, text_file_name, filenames):
  """Saves filenames to the given save directory as the given text file name.

  Args:
    save_dir: The directory to save the text file.
    text_file_name: Name of the text file to save.
    filenames: the array of file names to write on the text file.
  """

  text_file_path = os.path.join(save_dir, text_file_name)
  with open(text_file_path, 'w') as f:
    f.write('\n'.join(filenames))


def main(unused_argv):

  input_image_dir = os.path.join(FLAGS.dataset_dir, "Image")
  input_files = glob.glob(os.path.join(input_image_dir,
                                       '*.' + FLAGS.input_image_format))
  filenames = [os.path.splitext(os.path.basename(x))[0] for x in input_files]
  filenames = [x for x in filenames if not (x.lower().startswith("pacjent 6") or x.lower().startswith("pacjent 7") or x.lower().startswith("pacjent 2a"))]
  filenames = np.array(filenames)
  train_set, validation_set = _separate_sets(filenames)

  _write_to_text_file(FLAGS.dataset_dir, "trainval.txt", filenames)
  _write_to_text_file(FLAGS.dataset_dir, "train.txt", train_set)
  _write_to_text_file(FLAGS.dataset_dir, "val.txt", validation_set)


if __name__ == '__main__':
  tf.app.run()
