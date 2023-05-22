model_tests.py trains and tests a NN model that preddicts cluster membership.

From command line: python model_tests.py model [-e] [-f W01 W02 W03] [-r 42] [-t] [-bt] 
                                        [--min_n500 0] [--max_n500 60] [--min_z 0] [--max_z 1]

Arguments:
    model  .py file with model parameters (has to be in the same directory as model_tests.py). Without extension.
Options:
    -e     If model has already been saved and trained, load existing model and training history.
    -f     List of HSC fields (default: W01, W02, W03 & W04)
    -r     Change random_state for training-validation-test split (it's set to a fixed number by default)
    -t     Test different thresholds
    -bt    Compute thresholds that maximize F-Score or G-Means
    --min_n500, --max_n500    Minimum and maximum for cluster's n500. Default: None
    --min_z, --max_z          Minimum and maximum for cluster's z. Default: None


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



Directories in the metrics and saved model folders contain the results of various tests:
	1-test_architectures: tested 4 different models, with different numbers of layers and neurons, including dropout and gaussian dropout layers, and using the Adam and AdamW optimizers.
	2-test_loss: tested different loss functions.
	3-z_lims: trained and tested a model with galaxies around BCGs with different maximum redshifts (None, 1.0, 0.5, and 0.25).
	4-z-0.7: tested different architectures, training with galaxies around BCGs at up to z = 0.7
    5-z_steps: trained a model with BCGs in redshift ranges of size 0.1.

