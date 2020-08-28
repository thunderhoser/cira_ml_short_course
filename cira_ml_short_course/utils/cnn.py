"""Helper methods for CNNs (convolutional neural nets)."""

import copy
import random
import os.path
import dill
import numpy
import keras.models
from cira_ml_short_course.utils import utils, image_utils, \
    image_normalization, image_thresholding

KERNEL_INITIALIZER_NAME = 'glorot_uniform'
BIAS_INITIALIZER_NAME = 'zeros'

PLATEAU_PATIENCE_EPOCHS = 5
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.6
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 10
LOSS_PATIENCE = 0.

METRIC_FUNCTION_LIST = utils.METRIC_FUNCTION_LIST
METRIC_FUNCTION_DICT = utils.METRIC_FUNCTION_DICT

DEFAULT_INPUT_DIMENSIONS = numpy.array([32, 32, 4], dtype=int)
DEFAULT_CONV_BLOCK_LAYER_COUNTS = numpy.array([2, 2, 2, 2], dtype=int)
DEFAULT_CONV_CHANNEL_COUNTS = numpy.array(
    [32, 32, 64, 64, 128, 128, 256, 256], dtype=int
)
DEFAULT_CONV_DROPOUT_RATES = numpy.full(8, 0.)
DEFAULT_CONV_FILTER_SIZES = numpy.full(8, 3, dtype=int)
DEFAULT_DENSE_NEURON_COUNTS = numpy.array([776, 147, 28, 5, 1], dtype=int)
DEFAULT_DENSE_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, 0.5, 0])
DEFAULT_INNER_ACTIV_FUNCTION_NAME = copy.deepcopy(utils.RELU_FUNCTION_NAME)
DEFAULT_INNER_ACTIV_FUNCTION_ALPHA = 0.2
DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME = copy.deepcopy(utils.SIGMOID_FUNCTION_NAME)
DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA = 0.
DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

TRAINING_FILES_KEY = 'training_file_names'
VALIDATION_FILES_KEY = 'validation_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
NUM_EPOCHS_KEY = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'

METADATA_KEYS = [
    TRAINING_FILES_KEY, VALIDATION_FILES_KEY, NORMALIZATION_DICT_KEY,
    BINARIZATION_THRESHOLD_KEY, NUM_EPOCHS_KEY, NUM_EXAMPLES_PER_BATCH_KEY,
    NUM_TRAINING_BATCHES_KEY, NUM_VALIDATION_BATCHES_KEY
]


def _get_2d_conv_layer(
        num_rows_in_filter, num_columns_in_filter, num_rows_per_stride,
        num_columns_per_stride, num_filters, use_edge_padding=True,
        weight_regularizer=None):
    """Creates layer for 2-D convolution.

    :param num_rows_in_filter: Number of rows in each filter (kernel).
    :param num_columns_in_filter: Number of columns in each filter (kernel).
    :param num_rows_per_stride: Number of rows per filter stride.
    :param num_columns_per_stride: Number of columns per filter stride.
    :param num_filters: Number of filters (output channels).
    :param use_edge_padding: Boolean flag.  If True, output grid will be same
        size as input grid.  If False, output grid may be smaller.
    :param weight_regularizer: Will be used to regularize weights in the new
        layer.  This may be instance of `keras.regularizers` or None (if you
        want no regularization).
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    return keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(num_rows_in_filter, num_columns_in_filter),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding='same' if use_edge_padding else 'valid',
        dilation_rate=(1, 1), activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer
    )


def _get_2d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_rows_per_stride,
        num_columns_per_stride, do_max_pooling=True):
    """Creates layer for 2-D pooling.

    :param num_rows_in_window: Number of rows in pooling window.
    :param num_columns_in_window: Number of columns in pooling window.
    :param num_rows_per_stride: Number of rows per window stride.
    :param num_columns_per_stride: Number of columns per window stride.
    :param do_max_pooling: Boolean flag.  If True (False), will do max-
        (average-)pooling.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    if do_max_pooling:
        return keras.layers.MaxPooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(num_rows_per_stride, num_columns_per_stride),
            padding='valid'
        )

    return keras.layers.AveragePooling2D(
        pool_size=(num_rows_in_window, num_columns_in_window),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding='valid'
    )


def _write_metafile(
        dill_file_name, training_file_names, validation_file_names,
        normalization_dict, binarization_threshold, num_epochs,
        num_examples_per_batch, num_training_batches_per_epoch,
        num_validation_batches_per_epoch):
    """Writes metadata to Dill file.

    :param dill_file_name: Path to output file.
    :param training_file_names: See doc for `train_model_with_generator`.
    :param validation_file_names: Same.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    """

    metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        VALIDATION_FILES_KEY: validation_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        BINARIZATION_THRESHOLD_KEY: binarization_threshold,
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }

    utils._mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(metadata_dict, dill_file_handle)
    dill_file_handle.close()


def find_metafile(model_dir_name, raise_error_if_missing=True):
    """Finds metafile for CNN.

    :param model_dir_name: Name of model directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    metafile_name = '{0:s}/model_metadata.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def read_metafile(dill_file_name):
    """Reads metadata for neural net from Dill file.

    :param dill_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['training_file_names']: See doc for
        `train_model_with_generator`.
    metadata_dict['validation_file_names']: Same.
    metadata_dict['normalization_dict']: Same.
    metadata_dict['binarization_threshold']: Same.
    metadata_dict['num_epochs']: Same.
    metadata_dict['num_examples_per_batch']: Same.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    dill_file_handle = open(dill_file_name, 'rb')
    metadata_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)


def setup_cnn(
        input_dimensions=DEFAULT_INPUT_DIMENSIONS,
        conv_block_layer_counts=DEFAULT_CONV_BLOCK_LAYER_COUNTS,
        conv_layer_channel_counts=DEFAULT_CONV_CHANNEL_COUNTS,
        conv_layer_dropout_rates=DEFAULT_CONV_DROPOUT_RATES,
        conv_layer_filter_sizes=DEFAULT_CONV_FILTER_SIZES,
        dense_layer_neuron_counts=DEFAULT_DENSE_NEURON_COUNTS,
        dense_layer_dropout_rates=DEFAULT_DENSE_DROPOUT_RATES,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_batch_normalization=True):
    """Sets up (but does not train) CNN for binary classification.

    This method sets up the architecture, loss function, and optimizer.

    B = number of convolutional blocks
    C = number of convolutional layers
    D = number of dense layers

    :param input_dimensions: numpy array with dimensions of input data.  Entries
        should be (num_grid_rows, num_grid_columns, num_channels).
    :param conv_block_layer_counts: length-B numpy array with number of
        convolutional layers in each block.  Recall that each conv block except
        the last ends with a pooling layer.
    :param conv_layer_channel_counts: length-C numpy array with number of
        channels (filters) produced by each convolutional layer.
    :param conv_layer_dropout_rates: length-C numpy array of dropout rates.  To
        turn off dropout for a given layer, use NaN or a non-positive number.
    :param conv_layer_filter_sizes: length-C numpy array of filter sizes.  All
        filters will be square (num rows = num columns).
    :param dense_layer_neuron_counts: length-D numpy array with number of
        neurons for each dense layer.  The last value in this array is the
        number of target variables (predictands).
    :param dense_layer_dropout_rates: length-D numpy array of dropout rates.  To
        turn off dropout for a given layer, use NaN or a non-positive number.
    :param inner_activ_function_name: Name of activation function for all inner
        (non-output) layers.
    :param inner_activ_function_alpha: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    :param output_activ_function_name: Same as `inner_activ_function_name` but
        for output layer.
    :param output_activ_function_alpha: Same as `inner_activ_function_alpha` but
        for output layer.
    :param l1_weight: Weight for L_1 regularization.
    :param l2_weight: Weight for L_2 regularization.
    :param use_batch_normalization: Boolean flag.  If True, will use batch
        normalization after each inner layer.

    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    # TODO(thunderhoser): Allow for tasks other than binary classification.
    assert dense_layer_neuron_counts[-1] == 1

    num_conv_layers = len(conv_layer_channel_counts)
    assert numpy.sum(conv_block_layer_counts) == num_conv_layers

    num_input_rows = input_dimensions[0]
    num_input_columns = input_dimensions[1]
    num_input_channels = input_dimensions[2]

    input_layer_object = keras.layers.Input(
        shape=(num_input_rows, num_input_columns, num_input_channels)
    )
    regularizer_object = utils._get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    layer_object = None

    for i in range(num_conv_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = _get_2d_conv_layer(
            num_rows_in_filter=conv_layer_filter_sizes[i],
            num_columns_in_filter=conv_layer_filter_sizes[i],
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=conv_layer_channel_counts[i], use_edge_padding=True,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        layer_object = utils._get_activation_layer(
            function_name=inner_activ_function_name,
            slope_param=inner_activ_function_alpha
        )(layer_object)

        if conv_layer_dropout_rates[i] > 0:
            layer_object = utils._get_dropout_layer(
                dropout_fraction=conv_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization:
            layer_object = utils._get_batch_norm_layer()(layer_object)

        if i + 1 not in numpy.cumsum(conv_block_layer_counts):
            continue

        layer_object = _get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2, do_max_pooling=True
        )(layer_object)

    layer_object = keras.layers.Flatten()(layer_object)

    num_dense_layers = len(dense_layer_neuron_counts)

    for i in range(num_dense_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = utils._get_dense_layer(
            num_output_units=dense_layer_neuron_counts[i],
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_dense_layers - 1:
            layer_object = utils._get_activation_layer(
                function_name=output_activ_function_name,
                slope_param=output_activ_function_alpha
            )(layer_object)
        else:
            layer_object = utils._get_activation_layer(
                function_name=inner_activ_function_name,
                slope_param=inner_activ_function_alpha
            )(layer_object)

        if dense_layer_dropout_rates[i] > 0:
            layer_object = utils._get_dropout_layer(
                dropout_fraction=dense_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization and i != num_dense_layers - 1:
            layer_object = utils._get_batch_norm_layer()(layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )

    model_object.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=METRIC_FUNCTION_LIST
    )

    model_object.summary()
    return model_object


def create_data(image_file_names, normalization_dict, binarization_threshold):
    """Creates input data for CNN.

    This method is the same as `data_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param image_file_names: See doc for `data_generator`.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    :return: predictor_matrix: Same.
    :return: target_values: Same.
    :raises: TypeError: if `normalization_dict is None`.
    """

    image_dict = image_utils.read_many_files(image_file_names)

    predictor_matrix, _ = image_normalization.normalize_data(
        predictor_matrix=image_dict[image_utils.PREDICTOR_MATRIX_KEY],
        predictor_names=image_dict[image_utils.PREDICTOR_NAMES_KEY],
        normalization_dict=normalization_dict
    )

    target_values = image_thresholding.binarize_target_images(
        target_matrix=image_dict[image_utils.TARGET_MATRIX_KEY],
        binarization_threshold=binarization_threshold
    )

    print('Fraction of examples in positive class: {0:.4f}\n'.format(
        numpy.mean(target_values)
    ))

    return predictor_matrix, target_values


def data_generator(image_file_names, num_examples_per_batch, normalization_dict,
                   binarization_threshold):
    """Generates training or validation examples on the fly.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param image_file_names: 1-D list of paths to input files (readable by
        `image_utils.read_file`).
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: Dictionary with params used to normalize
        predictors.  See doc for `image_normalization.normalize_data`.
    :param binarization_threshold: Threshold used to binarize target variable.
        See doc for `image_thresholding.binarize_target_images`.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    # TODO(thunderhoser): Maybe add upsampling or downsampling.

    if normalization_dict is None:
        raise TypeError(
            'normalization_dict cannot be None.  Must be specified.'
        )

    random.shuffle(image_file_names)
    num_files = len(image_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                image_file_names[file_index]
            ))

            this_image_dict = image_utils.read_file(
                image_file_names[file_index]
            )
            predictor_names = this_image_dict[image_utils.PREDICTOR_NAMES_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[image_utils.PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = (
                    this_image_dict[image_utils.TARGET_MATRIX_KEY] + 0.
                )

            else:
                full_predictor_matrix = numpy.concatenate((
                    full_predictor_matrix,
                    this_image_dict[image_utils.PREDICTOR_MATRIX_KEY]
                ), axis=0)

                full_target_matrix = numpy.concatenate((
                    full_target_matrix,
                    this_image_dict[image_utils.TARGET_MATRIX_KEY]
                ), axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int
        )
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False
        )

        predictor_matrix, _ = image_normalization.normalize_data(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict
        )

        target_values = image_thresholding.binarize_target_images(
            target_matrix=full_target_matrix[batch_indices, ...],
            binarization_threshold=binarization_threshold
        )

        print('Fraction of examples in positive class: {0:.4f}\n'.format(
            numpy.mean(target_values)
        ))

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield predictor_matrix.astype('float32'), target_values


def train_model_with_generator(
        model_object, training_file_names, validation_file_names,
        normalization_dict, binarization_threshold, num_epochs,
        num_examples_per_batch, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, output_dir_name):
    """Trains CNN with generator.

    :param model_object: Untrained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param training_file_names: 1-D list of paths to training files (readable by
        `image_utils.read_file`).
    :param validation_file_names: Same but for validation files.
    :param normalization_dict: See doc for `data_generator`.
    :param binarization_threshold: Same.
    :param num_epochs: Number of epochs.
    :param num_examples_per_batch: Batch size.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param output_dir_name: Path to output directory (model will be saved here).
    """

    # TODO(thunderhoser): Make this method public.
    utils._mkdir_recursive_if_necessary(directory_name=output_dir_name)

    model_file_name = (
        output_dir_name + '/model_epoch={epoch:03d}_val-loss={val_loss:.6f}.h5'
    )

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=LOSS_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    metafile_name = find_metafile(output_dir_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        dill_file_name=metafile_name,
        training_file_names=training_file_names,
        validation_file_names=validation_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold,
        num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch
    )

    training_generator = data_generator(
        image_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )

    validation_generator = data_generator(
        image_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def train_model_sans_generator(
        model_object, training_file_names, validation_file_names,
        num_examples_per_batch, normalization_dict, binarization_threshold,
        num_epochs, output_dir_name):
    """Trains CNN without generator.

    :param model_object: See doc for `train_model_with_generator`.
    :param training_file_names: Same.
    :param validation_file_names: Same.
    :param num_examples_per_batch: Same.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    :param num_epochs: Same.
    :param output_dir_name: Same.
    """

    utils._mkdir_recursive_if_necessary(directory_name=output_dir_name)

    model_file_name = (
        output_dir_name + '/model_epoch={epoch:03d}_val-loss={val_loss:.6f}.h5'
    )

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=LOSS_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    metafile_name = find_metafile(output_dir_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        dill_file_name=metafile_name,
        training_file_names=training_file_names,
        validation_file_names=validation_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold,
        num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_training_batches_per_epoch=None,
        num_validation_batches_per_epoch=None
    )

    training_predictor_matrix, training_target_values = create_data(
        image_file_names=training_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )
    print('\n')

    validation_predictor_matrix, validation_target_values = create_data(
        image_file_names=validation_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )
    print('\n')

    model_object.fit(
        x=training_predictor_matrix, y=training_target_values,
        batch_size=num_examples_per_batch, epochs=num_epochs,
        steps_per_epoch=None, shuffle=True, verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=(validation_predictor_matrix, validation_target_values),
        validation_steps=None
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT
    )


def apply_model(model_object, predictor_matrix, verbose=True,
                output_layer_name=None):
    """Applies trained CNN to new data.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :param output_layer_name: Name of output layer.  If None, will use the
        actual output layer and return predicted probabilities.  If specified,
        will return "features" (outputs from the given layer).

    If `output_layer_name is None`...

    :return: forecast_probabilities: length-E numpy array with probabilities of
        positive class.

    If `output_layer_name` is specified...

    :return: feature_matrix: numpy array of features from the given layer.
        There is no guarantee on the shape of this array, except that the first
        axis has length E.
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 1000

    if output_layer_name is None:
        model_object_to_use = model_object
    else:
        model_object_to_use = keras.models.Model(
            inputs=model_object.input,
            outputs=model_object.get_layer(name=output_layer_name).output
        )

    output_array = None

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                this_first_index, this_last_index, num_examples
            ))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int
        )

        this_output_array = model_object_to_use.predict(
            predictor_matrix[these_indices, ...],
            batch_size=num_examples_per_batch
        )

        if output_layer_name is None:
            this_output_array = this_output_array[:, -1]

        if output_array is None:
            output_array = this_output_array + 0.
        else:
            output_array = numpy.concatenate(
                (output_array, this_output_array), axis=0
            )

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return output_array


def get_flattening_layer(cnn_model_object):
    """Finds flattening layer in CNN.

    This method assumes that there is only one flattening layer.  If there are
    several, this method will return the first (shallowest).

    :param cnn_model_object: Instance of `keras.models.Model`.
    :return: layer_name: Name of flattening layer.
    :raises: TypeError: if flattening layer cannot be found.
    """

    layer_names = [lyr.name for lyr in cnn_model_object.layers]

    flattening_flags = numpy.array(
        ['flatten' in n for n in layer_names], dtype=bool
    )
    flattening_indices = numpy.where(flattening_flags)[0]

    if len(flattening_indices) == 0:
        error_string = (
            'Cannot find flattening layer in model.  Layer names are listed '
            'below.\n{0:s}'
        ).format(str(layer_names))

        raise TypeError(error_string)

    return layer_names[flattening_indices[0]]
