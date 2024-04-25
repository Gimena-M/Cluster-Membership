"""
Class to perform random forest model tests (inherits from ModelTester).

Arguments for initialization are:
    * model: NN or RF model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.

The main() method performs the tests used by most scripts. Its arguments are:
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})
    * importances (list|None): List of importances to compute. If None, don't compute importances. (default: ['permutation_train', 'permutation_test', 'gini'])
    * sort_importances (str|None): Sort features in descending order of gini or permutation importance when plotting feature importances? Takes values None (no sorting), or importance name (default: 'gini')
    * permutation_train_max_samples (int|float): max_samples for permutation_importance with training sample (default: 1.0), 
    * permutation_test_max_samples (int|float): max_samples for permutation_importance with test sample (default: 1.0), 

File Saving:
    * Most plots are saved to a .png file.
    * Importances plots are saved to different .png files.
    * Other metrics (Loss, AUC, etc.), arguments used, and model summaries are saved to a .txt file.
    * Files are saved in a 'metrics' directory.
"""


from ModelTester import ModelTester

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class RFModelTester(ModelTester):

    plots = ["confusion matrix", "roc", "precision-recall"]
    rows: int = 1
    cols: int = 3
    figsize: tuple = (16,4)

    def main(self, optimize_threshold: bool = True, extra_args: dict = {}, 
              importances: list|None = ['permutation_train', 'permutation_test', 'gini'], sort_importances: str|None = 'gini', 
              permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):
        return super().main(optimize_threshold, extra_args, importances, sort_importances, permutation_train_max_samples, permutation_test_max_samples)
    
    def predict_score(self):
        scores = self.model.predict_proba(self.data.testing_features().values)  #probabilities
        self.scores = scores[:,1]

    def return_score(self, sample):
        s=  self.model.predict_proba(sample)
        return s[:,1]

    def write_report(self, extra_args: dict = {}):

        # write metrics to file: score, auc, classification report
        from sklearn.metrics import auc, classification_report
        model_score = self.model.score(self.data.testing_features(), self.data.testing_labels())

        with open(f'metrics/{self.name}.txt', mode='w') as file:

            for key in self.model.get_params():
                file.write(f'{key}: {self.model.get_params()[key]} \n')
            file.write('-'*70 + '\n')
            for key in self.data.args():
                file.write(f'{key}: {self.data.args()[key]} \n')
            for key in extra_args:
                file.write(f'{key}: {extra_args[key]} \n')
            file.write('-'*70 + '\n')
            for metric in self._metrics_report():
                    file.write(metric)

    def plot_importances(self, importances: list = ['permutation_train', 'permutation_test', 'gini'], sort_importances: str|None = 'gini', to_file: bool = True, permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):
        v = self.model.verbose
        self.model.verbose = 0
        super().plot_importances(importances, sort_importances, to_file, permutation_train_max_samples, permutation_test_max_samples)
        self.model.verbose = v
