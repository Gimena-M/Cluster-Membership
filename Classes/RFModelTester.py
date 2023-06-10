"""
Class to perform random forest model tests (inherits from ModelTester).

Arguments for initialization are:
    * model: NN or RF model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.

The main() method performs the tests used by most scripts. Its arguments are:
    * thresholds (list or None): List of thresholds to be tested. (default: None)
    * best_thresholds (bool): Compute thresholds that maximize F-Score or G-Means? (default: False)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})

File Saving:
    * Most plots are saved to a .png file.
    * Thresholds plots and importances plots are saved to different .png files.
    * Other metrics (Loss, AUC, etc.), arguments used, and model summaries are saved to a .txt file.
    * Files are saved in a 'metrics' directory.
"""


from ModelTester import ModelTester

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RFModelTester(ModelTester):

    plots = ["confusion matrix", "roc", "precision-recall"]
    rows: int = 1
    cols: int = 3
    figsize: tuple = (16,4)

    def main(self, thresholds: bool = False, best_thresholds: bool = False):
        super().main(thresholds, best_thresholds)
        self.plot_importances()

    def write_report(self):

        # write metrics to file: score, auc, classification report
        from sklearn.metrics import auc, classification_report
        model_score = self.model.score(self.data.testing_features(), self.data.testing_labels())

        with open(f'metrics/{self.name}.txt', mode='w') as file:

            for key in self.model.get_params():
                file.write(f'{key}: {self.model.get_params()[key]} \n')
            file.write('-'*70 + '\n')
            for key in self.data.args():
                file.write(f'{key}: {self.data.args()[key]} \n')
            file.write('-'*70 + '\n')
            file.write('Model score: {:.4g} \n'.format(model_score))
            file.write('ROC curve AUC: {}\n'.format(auc(self.fpr, self.tpr)))
            file.write('Precision-recall AUC: {}\n'.format(auc(self.rec, self.prec)))
            file.write('-'*70 + '\n')
            file.write(classification_report(self.data.testing_labels(),self.predictions))

    def plot_importances(self):
        # save plot of feature importances
        importances = np.sort(self.model.feature_importances_)[::-1]
        sorted_feat = [x for _,x in sorted(zip(self.model.feature_importances_, self.data.features))][::-1]

        plt.figure(figsize= (17, 10))
        plt.grid()
        sns.barplot(x = importances, y = sorted_feat)
        plt.savefig(f'metrics/importances_{self.name}.png', dpi=150, bbox_inches= 'tight')