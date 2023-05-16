rf.py trains and tests a Random Forest model for cluster membership prediction, and searches on model hyper parameters.

From command line: python rf.py [-m model.json] [-s search.json] [-r 42]
                                [--min_n500 0] [--max_n500 60] [--min_z 0] [--max_z 1]

Options:
    -m     Train and test a model with hyper parameters given in a JSON file. Save metrics.
    -s     Search on hyper parameters, with distributions given in a JSON file. Save metrics for best model.
    -r     Change random_state for training-testing split (it's set to a fixed number by default)
    --min_n500, --max_n500    Minimum and maximum for cluster's n500. Default: None
    --min_z, --max            Minimum and maximum for cluster's z. Default: None

Test results are saved into the "metrics" directory. Searches results are saved into "search_results".
The features to be used are listed in features.txt
JSON files have to be in the same directory as this script.

