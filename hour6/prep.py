import os
import csv
import json
import argparse
import tensorflow as tf

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def example(base_path, rel_path, labelidx, image_size=160):
    # get path
    image_path = os.path.join(base_path, rel_path)
    
    # load bits and resize
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    
    img_shape = img_final.shape
    assert img_shape[2] == 3, "Invalid channel count"
    
    # feature descriptions
    feature = {
        'height': _int64_feature(img_shape[0]),
        'width': _int64_feature(img_shape[1]),
        'depth': _int64_feature(img_shape[2]),
        'label': _int64_feature(int(labelidx)),
        'image': _floats_feature(img_final),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    return example

def main(data_path, output_path, target_output):
    info('Preprocess')
    fetch_step = os.path.join(output_path, 'fetch.json')
    print('Loading {}'.format(fetch_step))

    with open(fetch_step) as f:
        fetch = json.load(f)

    for i in fetch:
        print('{} => {}'.format(i, fetch[i]))

    categories = fetch['categories']
    index = fetch['index']
    raw_csv = os.path.join(output_path, fetch['file'])
    raw_folder = os.path.join(data_path, fetch['data'])

    info('Processing Images')    
    total_records = 0
    record_file = os.path.join(target_output, 'images.tfrecords')
    with open(raw_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        with tf.io.TFRecordWriter(record_file) as writer:
            for row in reader:
                try:
                    print('Trying {}...'.format(row[0]), end=' ')
                    image = example(raw_folder, row[0], row[2])
                    writer.write(image.SerializeToString())
                    total_records += 1
                    print('Success!')
                except Exception as e:
                    print('Error: {}'.format(e))

    print('Done!\nSaved to {}: processed {} records.'.format(record_file, total_records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data cleaning for binary image task')
    parser.add_argument('-d', '--data_path', help='directory to training data', default='data')
    parser.add_argument('-o', '--output_path', help='directory to previous data step', default='data')
    parser.add_argument('-t', '--target_output', help='target file to hold good data', default='data')
    args = parser.parse_args()

    params = vars(args)
    for i in params:
        print('{} => {}'.format(i, params[i]))

    main(**params)