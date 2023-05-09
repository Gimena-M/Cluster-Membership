import tensorflow as tf

layers = [
    # norm,
    tf.keras.layers.Dense(40, activation=tf.keras.activations.relu),
    # tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
    # tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]

compile_params = dict(
    optimizer = tf.keras.optimizers.experimental.AdamW(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
    )

normalization = True
epochs = 60
balance = 'weights'