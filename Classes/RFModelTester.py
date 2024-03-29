"""
Class to perform random forest model tests (inherits from ModelTester).

Arguments for initialization are:
    * model: NN or RF model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.

The main() method performs the tests used by most scripts. Its arguments are:
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})
    * sort_importances (str|None): Sort features in descending order of gini or permutation importance when plotting feature importances? Takes values None (no sorting), 'gini', and 'permutation' (default: 'permutation')

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

    def main(self, optimize_threshold: bool = True, extra_args: dict = {}, sort_importances: str|None = 'permutation', importances: bool = True):
        super().main(optimize_threshold, extra_args)
        if importances:
            self.plot_importances(sort_importances= sort_importances)

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

    def plot_importances(self, sort_importances: str|None = 'permutation', to_file: bool = True):
        # save plot of feature importances

        # compute permutation importance and gini importance
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(self.model, self.data.testing_features(), self.data.testing_labels(), scoring= ['average_precision'], n_jobs= 1, n_repeats = 3)
        perm = pi['average_precision'].importances_mean
        gini = self.model.feature_importances_

        importances = pd.DataFrame(dict(permutation = perm, gini = gini, feature = self.data.features))
        importances.to_csv(f'metrics/importances_{self.name}.csv', index= False)
        if sort_importances:
            importances = importances.sort_values(by = sort_importances, ascending= False)
        importances = pd.melt(importances, var_name="type", value_name="importance", id_vars= 'feature')

        plt.figure(figsize= (10, len(self.data.features)/3.))
        sns.barplot(data = importances, x = 'importance', y = 'feature', hue = 'type', orient = 'h')
        plt.grid()
        if to_file:
            plt.savefig(f'metrics/importances_{self.name}.png', dpi=150, bbox_inches= 'tight')
            plt.close()   
        else: 
            plt.show()     

        # if sort_importances:
        #     importances = np.sort(self.model.feature_importances_)[::-1]
        #     sorted_feat = [x for _,x in sorted(zip(self.model.feature_importances_, self.data.features))][::-1]
        #     sns.barplot(x = importances, y = sorted_feat)
        # else:
        #     sns.barplot(x = self.model.feature_importances_, y = self.data.features)