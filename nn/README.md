This directory contains some tests:

+ 1-test_architectures: tested different models, with different numbers of layers and neurons, including dropout and gaussian dropout layers.
+ 2-test_optimizer: tested different optimizers.
+ 3-test_loss: tested different loss functions.
+ 4-z_lims: trained and tested a model with galaxies around BCGs with different maximum redshifts None, 1.0, 0.5, and 0.25.
+ 5-z-0.7: tested different architectures, training with galaxies around BCGs at up to z = 0.7
+ 6-z_steps: trained a model with BCGs in redshift ranges of size 0.1.
+ 7-mag_i_lims: trained a model with galaxies with i magnitude lower than 19, 21, and with no limit.
+ 8-filtered_z: compared performance using tables with all galaxies, and tables with only galaxies within a z range around the nearest BCG ("z_filtered")
+ 9-features: compared performance using different sets of features.
+ 10-mass_lims: compared performance using all galaxies and using only galaxies with log_st_mass > 10.5

Each directory contains:

+ A python script
+ A "metrics" folder with results.
+ A "saved_models" folder with models.

A basic script would be:

    import sys
    import tensorflow as tf

    # Imports from Classes directory
    sys.path.append('../../Classes')
    from DataHandler import DataHandler
    from NNModelController import NNModelController

    # Architecture
    layers = [
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    name = "test"

    # Compile parameters:
    compile_params = dict(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[]   
    )

    # Read and prepare data, train and test model
    data = DataHandler(validation_sample = True, features_txt = 'all_features.txt', fields_list = ['W01', 'W02', 'W03', 'W04'], balance = 'weights')
    mod = NNModelController (compile_params= compile_params, data= data, name= name, layers= layers)
    mod.main(read_data = True)

The "NOTEBOOKS" directory has jupyter notebooks with other smaller tests.