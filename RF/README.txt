rf.py trains and tests a Random Forest model for cluster membership prediction, and searches on model hyper parameters.

From command line: python rf.py [-m model.json] [-s search.json] [-l model.joblib] [-f W01 W02 W03] [-r 42]
                                [--min_n500 0] [--max_n500 60] [--min_z 0] [--max_z 1]

Options:
    -m     Train and test a model with hyper parameters given in a JSON file. Save metrics.
    -s     Search on hyper parameters, with distributions given in a JSON file. Save metrics for best model.
    -l     Load existing model.
    -f     List of HSC fields (default: W01, W02, W03 & W04)
    -r     Change random_state for training-testing split (it's set to a fixed number by default)
    --min_n500, --max_n500    Minimum and maximum for cluster's n500. Default: None
    --min_z, --max            Minimum and maximum for cluster's z. Default: None

Test results are saved into the "metrics" directory. Searches results are saved into "search_results".
The features to be used are listed in features.txt
JSON files have to be in the same directory as this script.



Directories in the metrics and saved model folders contain the results of various tests:
	1_model-1: tested one model with parameters determined from the results of a random search (saved in the search_results directory)
    2_z-steps-0.1: trained a model with BCGs in redshift ranges of size 0.1.
    3_z-steps-0.05: trained a model with BCGs in redshift ranges of size 0.05.