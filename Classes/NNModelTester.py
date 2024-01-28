"""
Class to perform neural network model tests (inherits from ModelTester).

Arguments for initialization are:
    * model: NN model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.
    * history (dict): Training history.

The main() method performs the tests used by most scripts. Its arguments are:
    * thresholds (list or None): List of thresholds to be tested. (default: None)
    * best_thresholds (bool): Compute thresholds that maximize F-Score or G-Means? (default: False)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})
    * loss_lims (tuple): y limits for a loss vs epochs plot (for neural networks). (default: (None, None))

File Saving:
    * Most plots are saved to a .png file.
    * Thresholds plots and importances plots are saved to different .png files.
    * Other metrics (Loss, AUC, etc.), arguments used, and model summaries are saved to a .txt file.
    * Files are saved in a 'metrics' directory.
"""


from DataHandler import DataHandler
from ModelTester import ModelTester

import numpy as np

class NNModelTester(ModelTester):

    plots = ["loss", "confusion matrix", "roc", "precision-recall"]
    rows: int = 2
    cols: int = 2
    figsize: tuple = (10,10)

    def __init__(self, model, data: DataHandler, name: str, history: dict):
        super().__init__(model, data, name)
        self.history = history

    def predict(self):
        self.scores = self.model.predict(self.data.testing_features(), verbose = 0)  #probabilities
        self.predictions = np.round(self.scores, decimals = 0)      #labels

    def curves(self):
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
        self.fpr, self.tpr, self.thres_roc = roc_curve(self.data.testing_labels(), self.scores, pos_label=1)
        self.prec, self.rec, self.thres_pr = precision_recall_curve(self.data.testing_labels(), self.scores, pos_label= 1)

        self.test_loss = self.model.evaluate(self.data.testing_features(), self.data.testing_labels(), verbose=0)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.pr_auc = auc(self.rec, self.prec)
        self.f1 = f1_score(self.data.testing_labels(), self.predictions)

    def write_report(self, extra_args: dict = {}, to_file = True):

        # write metrics into file: loss, auc, classification report
        # write also a model summary, arguments of the DataHandler, and extra args.
        
        from sklearn.metrics import classification_report

        if to_file:
            def model_write(string):
                file.write(string + '\n')

            with open(f'metrics/{self.name}.txt', mode='w') as file:
                self.model.summary(print_fn= model_write)
                file.write('\n\n')
                
                for key in self.data.args():
                    file.write(f'{key}: {self.data.args()[key]} \n')
                file.write('-'*70 + '\n')
                
                for key in extra_args:
                    file.write(f'{key}: {extra_args[key]} \n')
                file.write('-'*70 + '\n')
                
                try:
                    file.write('Optimizer: {} \n'.format(self.model.optimizer._name))
                except:
                    file.write('Optimizer: {} \n'.format(self.model.optimizer.name))

                try:
                    file.write('Loss function: {} \n'.format(self.model.loss.name))
                except:
                    file.write('Loss function: {} \n'.format(self.model.loss._name))
                    
                file.write('-'*70 + '\n')
                file.write('Loss on test dataset: {:.4g} \n'.format(self.test_loss))
                file.write('ROC curve AUC: {}\n'.format(self.roc_auc))
                file.write('Precision-recall AUC: {}\n'.format(self.pr_auc))
                file.write('-'*70 + '\n')
                file.write(classification_report(self.data.testing_labels(),self.predictions))
        else:
            print(self.model.summary())
            print('\n\n')
                
            for key in self.data.args():
                print(f'{key}: {self.data.args()[key]} \n')
            print('-'*70 + '\n')
            
            for key in extra_args:
               print(f'{key}: {extra_args[key]} \n')
            print('-'*70 + '\n')
            
            try:
               print('Optimizer: {} \n'.format(self.model.optimizer._name))
            except:
                print('Optimizer: {} \n'.format(self.model.optimizer.name))

            try:
                print('Loss function: {} \n'.format(self.model.loss.name))
            except:
                print('Loss function: {} \n'.format(self.model.loss._name))
                
            print('-'*70 + '\n')
            print('Loss on test dataset: {:.4g} \n'.format(self.test_loss))
            print('ROC curve AUC: {}\n'.format(self.roc_auc))
            print('Precision-recall AUC: {}\n'.format(self.pr_auc))
            print('-'*70 + '\n')
            print(classification_report(self.data.testing_labels(),self.predictions))