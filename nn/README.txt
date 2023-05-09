model_tests.py trains and tests a NN model that preddicts cluster membership.

From command line: python model_tests.py model.py [-e] [-r 42] [-t] [-bt] 
Arguments:
    .py file with model parameters (has to be in the same directory as model_tests.py).
Options:
    -e     If model has already been saved and trained, load existing model and training history.
    -r     Change random_state for training-validation-split (it's set to a fixed number by default)
    -t     Test different thresholds
    -bt    Compute thresholds that maximize F-Score or G-Means

The .py file with model parameters has:
    layers: list of layers for the network
    compile_params: parameters for model.compile(). Optimizer, loss function, metrics...
    epochs: number of epochs
    normalization: if True, a normalization layer is added as a first layer, and adapted with features.
    balance: how to deal with class imbalance. Can be 'weights', 'SMOTE' or None
There's a bunch of models in the "models_scripts" directory.

Results are saved into a "metrics" directory. For each model, there's an image with plots and a text file with metrics.
The features to be used are listed in features.txt

Models are saved into the "saved_models" directory.

The "NOTEBOOKS" directory has jupyter notebooks with other smaller tests.

