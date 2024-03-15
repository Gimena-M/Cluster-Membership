"""
Class to perform random forest model tests (inherits from ModelTester).

Arguments for initialization are:
    * model: NN or RF model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.

The main() method performs the tests used by most scripts. Its arguments are:
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})
    * sort_importances (bool): Sort features in descending order of importance when plotting feature importances? (default: True)

File Saving:
    * Most plots are saved to a .png file.
    * Importances plots are saved to different .png files.
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

    def main(self, optimize_threshold: bool = True, extra_args: dict = {}, sort_importances: bool = True):
        super().main(optimize_threshold, extra_args)
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
            file.write('Model score: {:.4g} \n'.format(model_score))
            file.write('ROC curve AUC: {}\n'.format(auc(self.fpr, self.tpr)))
            file.write('Precision-recall AUC: {}\n'.format(auc(self.rec, self.prec)))
            file.write('-'*70 + '\n')
            file.write(classification_report(self.data.testing_labels(),self.predictions))

    def plot_importances(self, sort_importances: bool = True, to_file: bool = True):
        # save plot of feature importances

        plt.figure(figsize= (16, 18))

        if sort_importances:
            importances = np.sort(self.model.feature_importances_)[::-1]
            sorted_feat = [x for _,x in sorted(zip(self.model.feature_importances_, self.data.features))][::-1]
            sns.barplot(x = importances, y = sorted_feat)
        else:
            sns.barplot(x = self.model.feature_importances_, y = self.data.features)
       
        plt.grid()
        if to_file:
            plt.savefig(f'metrics/importances_{self.name}.png', dpi=150, bbox_inches= 'tight')
            plt.close()   
        else: 
            plt.show()     