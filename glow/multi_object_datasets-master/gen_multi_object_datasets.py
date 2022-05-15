
import multi_dsprites

tf_records_path = '/home/thu/dataset/multi_object_datasets/tetrominoes_tetrominoes_train.tfrecords'
batch_size = 32

if False:
  dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
  ds_numpy = tfds.as_numpy(dataset)
  print(ds_numpy)
  for data in ds_numpy:
    print(data)

if True:
  dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
  dataset = dataset.batch(1)
  for data in dataset:
    print(data)


def input_fn(filename):

  return dataset


#estimator.train(input_fn=lambda: input_fn())