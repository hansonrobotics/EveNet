# CsvReader without gc implementation.
import os
import re
import fnmatch
import threading
import tensorflow as tf
import multiprocessing


class CsvReader(object):
    def __init__(self, files, batch_size, receptive_field, sample_size, config):
        # indicates one chunk of data.
        chunk_size = receptive_field + sample_size

        self.data_dim = config["data_dim"]

        # Initialize the main data batch. This uses raw values, no lookup table.
        data_files = [files[i] for i in range(len(files)) if files[i].endswith(config["data_suffix"])]

        self.data_batch = self.input_batch(data_files, config["data_dim"], batch_size=batch_size, chunk_size=chunk_size,skip_header_lines=1)

        emotions_files = []
        for df in data_files:
            em_file = df[:-3]+"emo"
            emotions_files.append(em_file)
        phonemes_files = []
        for df in data_files:
            phone_file = df[:-3]+"pho"
            phonemes_files.append(phone_file)
        
        if config["emotion_enabled"]:
            emotion_dim = config["emotion_dim"]
            emotion_categories = config["emotion_categories"]
            
            self.emotion_cardinality = len(emotion_categories)
            self.gc_batch = self.input_batch(emotions_files,
                                             emotion_dim,
                                             batch_size=batch_size,
                                             chunk_size=chunk_size,
                                             mapping_strings=emotion_categories)
        else:
            self.gc_batch = None
        if config["phoneme_enabled"]:
            phoneme_dim = config["phoneme_dim"]
            phoneme_categories = config["phoneme_categories"]
            
            
            self.phoneme_cardinality = len(phoneme_categories)
            self.lc_batch = self.input_batch(phonemes_files,
                                             phoneme_dim,
                                             batch_size=batch_size,
                                             chunk_size=chunk_size,
                                             mapping_strings=phoneme_categories)
        else:
            self.lc_batch = None

    def input_batch(self,
                    filenames,
                    data_dim,
                    num_epochs=None,
                    skip_header_lines=0,
                    batch_size=10,
                    chunk_size=100,
                    mapping_strings=None):
                    
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

        _, rows = reader.read_up_to(filename_queue, num_records=chunk_size)

        # Parse the CSV File
        if mapping_strings:
            default_value = "N/A"
        else:
            default_value = 1.0

        record_defaults = [[default_value] for _ in range(data_dim)]
        features = tf.decode_csv(rows, record_defaults=record_defaults)

        # Mapping for Conditioning Files, replace String by lookup table.
        if mapping_strings:
            table = tf.contrib.lookup.index_table_from_tensor(tf.constant(mapping_strings))
            features = table.lookup(tf.stack(features))
            features = tf.unstack(features)

        # Resize the feature vector in case we got an incomplete read from the reader
        features = tf.transpose(features)
        features = tf.image.resize_images(tf.reshape(features, [-1, data_dim, 1]),
                                          size=[chunk_size, data_dim],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        features = tf.reshape(features, [chunk_size, data_dim])

        # WARNING...
        features = tf.train.batch(
            [features],
            batch_size,
            shapes=[chunk_size, data_dim],
            capacity=batch_size * 100,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=False,
            allow_smaller_final_batch=False
        )

        return features
