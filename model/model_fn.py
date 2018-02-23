"""Define the model."""

import tensorflow as tf
def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out  = inputs["images"]
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
    weight={}
    cold=1
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            weight["w{0}".format(i)] = tf.get_variable("W{0}".format(i), [3,3,cold,c], initializer = tf.contrib.layers.xavier_initializer(seed=0))
            newout = tf.nn.conv2d(out, weight["w{0}".format(i)], strides=[1,1,1,1], padding='SAME')
#            if params.use_batch_norm:

  #              out = batch_norm_wrapper(out, is_training, decay = 0.999,epsilon=1E-8)
            cold=c
            out = tf.nn.relu(newout)
    out = tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'VALID')


    #out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
    with tf.variable_scope('fc_1'):
        out = tf.contrib.layers.flatten(out)
#        if params.use_batch_norm:
   #         out = batch_norm_wrapper(out, is_training,  decay = 0.999,epsilon=1E-8)
        out = tf.nn.relu(out)
    out = tf.contrib.layers.flatten(out)
    with tf.variable_scope('fc_2'):
        logits = tf.contrib.layers.fully_connected(out, num_outputs=params.num_labels,activation_fn=None)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs["labels"]
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)
    # Define loss and accuracy
    label_argmax = tf.argmax(labels,1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    loss = tf.cast(loss, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label_argmax,predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.contrib.slim.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.contrib.metrics.streaming_accuracy(labels=label_argmax, predictions=tf.argmax(logits, 1)),
            'loss': tf.contrib.metrics.streaming_mean(loss)
        }
    chto_hand = []
    for item in metrics.values():
        chto_hand.append(item[1])
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*chto_hand)

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(label_argmax, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label_argmax))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format("label_argmax"), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
