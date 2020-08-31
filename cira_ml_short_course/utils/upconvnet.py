"""Helper methods for upconvnets (upconvolutional networks)."""

import copy
import numpy
import keras.models
from cira_ml_short_course.utils import cnn, utils, image_utils, \
    image_normalization

KERNEL_INITIALIZER_NAME = cnn.KERNEL_INITIALIZER_NAME
BIAS_INITIALIZER_NAME = cnn.BIAS_INITIALIZER_NAME

PLATEAU_PATIENCE_EPOCHS = cnn.PLATEAU_PATIENCE_EPOCHS
PLATEAU_LEARNING_RATE_MULTIPLIER = cnn.PLATEAU_LEARNING_RATE_MULTIPLIER
PLATEAU_COOLDOWN_EPOCHS = cnn.PLATEAU_COOLDOWN_EPOCHS
EARLY_STOPPING_PATIENCE_EPOCHS = cnn.EARLY_STOPPING_PATIENCE_EPOCHS
LOSS_PATIENCE = cnn.LOSS_PATIENCE

DEFAULT_INPUT_DIMENSIONS = numpy.array([4, 4, 256], dtype=int)
DEFAULT_CONV_BLOCK_LAYER_COUNTS = numpy.array([2, 2, 2, 2], dtype=int)
DEFAULT_CONV_CHANNEL_COUNTS = numpy.array(
    [256, 128, 128, 64, 64, 32, 32, 4], dtype=int
)
DEFAULT_CONV_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0])
DEFAULT_CONV_FILTER_SIZES = numpy.full(8, 3, dtype=int)
DEFAULT_INNER_ACTIV_FUNCTION_NAME = copy.deepcopy(utils.RELU_FUNCTION_NAME)
DEFAULT_INNER_ACTIV_FUNCTION_ALPHA = 0.2
DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME = None
DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA = 0.
DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001


def _get_transposed_conv_layer(
        num_rows_in_filter, num_columns_in_filter, upsampling_factor,
        num_filters, weight_regularizer=None):
    """Creates layer for 2-D transposed convolution.

    :param num_rows_in_filter: Number of rows in each filter (kernel).
    :param num_columns_in_filter: Number of columns in each filter (kernel).
    :param upsampling_factor: Upsampling factor (integer >= 1).
    :param num_filters: Number of filters (output channels).
    :param weight_regularizer: Will be used to regularize weights in the new
        layer.  This may be instance of `keras.regularizers` or None (if you
        want no regularization).
    :return: layer_object: Instance of `keras.layers.Conv2DTranspose`.
    """

    return keras.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=(num_rows_in_filter, num_columns_in_filter),
        strides=(upsampling_factor, upsampling_factor),
        padding='same',
        dilation_rate=(1, 1), activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer
    )


def _get_upsampling_layer(upsampling_factor):
    """Creates layer for 2-D upsampling.

    :param upsampling_factor: Upsampling factor (integer >= 1).
    :return: layer_object: Instance of `keras.layers.Upsampling2D`.
    """

    try:
        return keras.layers.UpSampling2D(
            size=(upsampling_factor, upsampling_factor),
            data_format='channels_last', interpolation='bilinear'
        )
    except:
        return keras.layers.UpSampling2D(
            size=(upsampling_factor, upsampling_factor),
            data_format='channels_last'
        )


def setup_upconvnet(
        input_dimensions=DEFAULT_INPUT_DIMENSIONS,
        conv_block_layer_counts=DEFAULT_CONV_BLOCK_LAYER_COUNTS,
        conv_layer_channel_counts=DEFAULT_CONV_CHANNEL_COUNTS,
        conv_layer_dropout_rates=DEFAULT_CONV_DROPOUT_RATES,
        conv_layer_filter_sizes=DEFAULT_CONV_FILTER_SIZES,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_transposed_conv=True, use_batch_norm_inner=True,
        use_batch_norm_output=True):
    """Sets up (but does not train) upconvnet.

    This method sets up the architecture, loss function, and optimizer.

    B = number of convolutional blocks
    C = number of convolutional layers
    D = number of dense layers

    :param input_dimensions: numpy array with dimensions of input data.  Entries
        should be (num_grid_rows, num_grid_columns, num_channels).
    :param conv_block_layer_counts: length-B numpy array with number of
        convolutional layers in each block.  Remember that each conv block
        except the last upsamples the image by a factor of 2.
    :param conv_layer_channel_counts: length-C numpy array with number of
        channels (filters) produced by each convolutional layer.
    :param conv_layer_dropout_rates: length-C numpy array of dropout rates.  To
        turn off dropout for a given layer, use NaN or a non-positive number.
    :param conv_layer_filter_sizes: length-C numpy array of filter sizes.  All
        filters will be square (num rows = num columns).
    :param inner_activ_function_name: Name of activation function for all inner
        (non-output) layers.
    :param inner_activ_function_alpha: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    :param output_activ_function_name: Same as `inner_activ_function_name` but
        for output layer.  This may be None.
    :param output_activ_function_alpha: Same as `inner_activ_function_alpha` but
        for output layer.
    :param l1_weight: Weight for L_1 regularization.
    :param l2_weight: Weight for L_2 regularization.
    :param use_transposed_conv: Boolean flag.  If True (False), will use
        transposed convolution (upsampling followed by normal convolution).
    :param use_batch_norm_inner: Boolean flag.  If True, will use batch
        normalization after each inner layer.
    :param use_batch_norm_output: Same but for output layer.

    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    num_conv_layers = len(conv_layer_channel_counts)
    assert numpy.sum(conv_block_layer_counts) == num_conv_layers

    num_input_rows = input_dimensions[0]
    num_input_columns = input_dimensions[1]
    num_input_channels = input_dimensions[2]

    input_layer_object = keras.layers.Input(
        shape=(numpy.prod(input_dimensions),)
    )
    regularizer_object = utils._get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    layer_object = keras.layers.Reshape(
        target_shape=(num_input_rows, num_input_columns, num_input_channels)
    )(input_layer_object)

    for i in range(num_conv_layers):
        if (
                i + 1 in numpy.cumsum(conv_block_layer_counts)
                and i != num_conv_layers - 1
        ):
            if use_transposed_conv:
                layer_object = _get_transposed_conv_layer(
                    num_rows_in_filter=conv_layer_filter_sizes[i],
                    num_columns_in_filter=conv_layer_filter_sizes[i],
                    upsampling_factor=2,
                    num_filters=conv_layer_channel_counts[i],
                    weight_regularizer=regularizer_object
                )(layer_object)
            else:
                layer_object = _get_upsampling_layer(
                    upsampling_factor=2
                )(layer_object)

                layer_object = cnn._get_2d_conv_layer(
                    num_rows_in_filter=conv_layer_filter_sizes[i],
                    num_columns_in_filter=conv_layer_filter_sizes[i],
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=conv_layer_channel_counts[i],
                    use_edge_padding=True,
                    weight_regularizer=regularizer_object
                )(layer_object)
        else:
            layer_object = cnn._get_2d_conv_layer(
                num_rows_in_filter=conv_layer_filter_sizes[i],
                num_columns_in_filter=conv_layer_filter_sizes[i],
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=conv_layer_channel_counts[i], use_edge_padding=True,
                weight_regularizer=regularizer_object
            )(layer_object)

        if i == num_conv_layers - 1:
            if output_activ_function_name is not None:
                layer_object = utils._get_activation_layer(
                    function_name=output_activ_function_name,
                    slope_param=output_activ_function_alpha
                )(layer_object)
        else:
            layer_object = utils._get_activation_layer(
                function_name=inner_activ_function_name,
                slope_param=inner_activ_function_alpha
            )(layer_object)

        if conv_layer_dropout_rates[i] > 0:
            layer_object = utils._get_dropout_layer(
                dropout_fraction=conv_layer_dropout_rates[i]
            )(layer_object)

        if i != num_conv_layers - 1 and use_batch_norm_inner:
            layer_object = utils._get_batch_norm_layer()(layer_object)

        if i == num_conv_layers - 1 and use_batch_norm_output:
            layer_object = utils._get_batch_norm_layer()(layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )

    model_object.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam()
    )

    model_object.summary()
    return model_object


def create_data(image_file_names, normalization_dict, cnn_model_object):
    """Creates input data for upconvnet.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    Z = number of features (from CNN's flattening layer)

    :param image_file_names: 1-D list of paths to input files (readable by
        `image_utils.read_file`).
    :param normalization_dict: Dictionary with params used to normalize
        predictors.  See doc for `image_normalization.normalize_data`.
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).  Inputs for upconvnet will be outputs from
        CNN's flattening layer.
    :return: feature_matrix: E-by-Z numpy array of features.  These are inputs
        for the upconvnet.
    :return: target_matrix: E-by-M-by-N-by-C numpy array of target values.
        These are targets for the upconvnet but inputs for the CNN.
    """

    image_dict = image_utils.read_many_files(image_file_names)

    target_matrix, _ = image_normalization.normalize_data(
        predictor_matrix=image_dict[image_utils.PREDICTOR_MATRIX_KEY],
        predictor_names=image_dict[image_utils.PREDICTOR_NAMES_KEY],
        normalization_dict=normalization_dict
    )

    feature_matrix = cnn.apply_model(
        model_object=cnn_model_object, predictor_matrix=target_matrix,
        verbose=True,
        output_layer_name=cnn.get_flattening_layer(cnn_model_object)
    )

    return feature_matrix, target_matrix


def train_model_sans_generator(
        model_object, cnn_model_object, training_file_names,
        validation_file_names, num_examples_per_batch, normalization_dict,
        num_epochs, output_dir_name):
    """Trains upconvnet without generator.

    :param model_object: Untrained upconvnet (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param training_file_names: 1-D list of paths to training files (readable by
        `image_utils.read_file`).
    :param validation_file_names: Same but for validation files.
    :param num_examples_per_batch: Batch size.
    :param normalization_dict: See doc for `create_data`.
    :param num_epochs: Number of epochs.
    :param output_dir_name: Path to output directory (model will be saved here).
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

    training_feature_matrix, training_target_matrix = create_data(
        image_file_names=training_file_names,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object
    )
    print('\n')

    validation_feature_matrix, validation_target_matrix = create_data(
        image_file_names=validation_file_names,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object
    )
    print('\n')

    model_object.fit(
        x=training_feature_matrix, y=training_target_matrix,
        batch_size=num_examples_per_batch, epochs=num_epochs,
        steps_per_epoch=None, shuffle=True, verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=(validation_feature_matrix, validation_target_matrix),
        validation_steps=None
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    """

    return keras.models.load_model(hdf5_file_name)


def apply_model(model_object, cnn_model_object, cnn_predictor_matrix,
                verbose=True):
    """Applies trained upconvnet to new data.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param model_object: Trained upconvnet (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param cnn_predictor_matrix: E-by-M-by-N-by-C numpy array of predictor
        values for CNN.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: reconstructed_predictor_matrix: Upconvnet reconstruction of
        `cnn_predictor_matrix`.
    """

    num_examples = cnn_predictor_matrix.shape[0]
    num_examples_per_batch = 1000
    reconstructed_predictor_matrix = numpy.full(
        cnn_predictor_matrix.shape, numpy.nan
    )

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print((
                'Applying upconvnet to examples {0:d}-{1:d} of {2:d}...'
            ).format(
                this_first_index, this_last_index, num_examples
            ))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int
        )

        this_feature_matrix = cnn.apply_model(
            model_object=cnn_model_object,
            predictor_matrix=cnn_predictor_matrix[these_indices, ...],
            verbose=False,
            output_layer_name=cnn.get_flattening_layer(cnn_model_object)
        )

        reconstructed_predictor_matrix[these_indices, ...] = (
            model_object.predict(
                this_feature_matrix, batch_size=len(these_indices)
            )
        )

    if verbose:
        print('Have applied upconvnet to all {0:d} examples!'.format(
            num_examples
        ))

    return reconstructed_predictor_matrix
