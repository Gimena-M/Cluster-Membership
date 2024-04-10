"""
Class to perform neural network model tests (inherits from ModelTester).

Arguments for initialization are:
    * model: NN model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.
    * history (dict): Training history.

The main() method performs the tests used by most scripts. Its arguments are:
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})
    * loss_lims (tuple): y limits for a loss vs epochs plot (for neural networks). (default: (None, None))

File Saving:
    * Plots are saved to a .png file.
    * Other metrics (Loss, AUC, etc.), arguments used, and model summaries are saved to a .txt file.
    * Files are saved in a 'metrics' directory.
"""


from pandas import DataFrame
from DataHandler import DataHandler
from ModelTester import ModelTester

class NNModelTester(ModelTester):

    plots = ["loss", "confusion matrix", "roc", "precision-recall"]
    rows: int = 2
    cols: int = 2
    figsize: tuple = (10,10)

    def __init__(self, model, data: DataHandler, name: str, history: dict = {}):
        super().__init__(model, data, name)
        self.history = history

    def main(self, optimize_threshold: bool = True, extra_args: dict = {}, 
              importances: list|None = ['permutation_train', 'permutation_test'], sort_importances: str|None = 'permutation_train', 
              permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0, loss_lims: tuple = (None, None)):
        return super().main(optimize_threshold, extra_args, importances, sort_importances, permutation_train_max_samples, permutation_test_max_samples, loss_lims)

    def predict_score(self):
        self.scores = self.model.predict(self.data.testing_features(), verbose = 0)  #probabilities

    def compute_metrics(self):
        super().compute_metrics()
        self.test_loss = self.model.evaluate(self.data.testing_features(), self.data.testing_labels(), verbose=0)

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
                for metric in self._metrics_report():
                    file.write(metric)
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
            for metric in self._metrics_report():
                    print(metric)


    def plot_importances(self, importances: list = ['permutation_train', 'permutation_test'], sort_importances: str|None = 'permutation_train', to_file: bool = True, permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):
        return super().plot_importances(importances, sort_importances, to_file, permutation_train_max_samples, permutation_test_max_samples)

    def return_score(self, sample: DataFrame):
        return self.model.predict(sample, verbose = 0)