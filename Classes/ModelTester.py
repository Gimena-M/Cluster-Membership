"""
Superclass to perform model tests (inherits to RFModelTester and NNModelTester). 
It is not meant to be instantiated.

Arguments for initialization are:
    * model: NN or RF model to be tested.
    * data (DataHandler): Instance of DataHandler.
    * name (str): Name to be used on file names that save metrics.

The main() method performs the tests used by most scripts. Its arguments are:
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * extra_args (dict): Extra arguments to be saved in a txt file along with metrics. (default: {})
    * loss_lims (tuple): y limits for a loss vs epochs plot (for neural networks). (default: (None, None))

File Saving:
    * Most plots are saved to a .png file.
    * Importance plots are saved to different .png files.
    * Other metrics (Loss, AUC, etc.), arguments used, and model summaries are saved to a .txt file.
    * Files are saved in a 'metrics' directory.
"""

from DataHandler import DataHandler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

class ModelTester:

    plots = [] # can include 'confusion_matrix', 'roc', 'precision_recall', 'loss'
    rows: int = 0 # number of rows for plt.subplot()
    cols: int = 0 # number of columns for plt.subplot()
    figsize: tuple = (0,0) # figure size

    def __init__(self, model, data: DataHandler, name: str):
        self.model = model
        self.data = data
        self.name = name
        self.threshold = 0.5
        # self.history = history

    def main(self, optimize_threshold: bool = True, extra_args: dict = {}, importances: list|None = None, sort_importances: str|None = None, permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0, loss_lims: tuple = (None, None)):
        
        self.predict(optimize_threshold)
        self.write_report(extra_args)
        self.plot_report(loss_lims= loss_lims)
        if importances:
            self.plot_importances(importances= importances, sort_importances= sort_importances, permutation_train_max_samples= permutation_train_max_samples, permutation_test_max_samples= permutation_test_max_samples)

    def predict(self, optimize_threshold: bool = True):
        self.predict_score()
        self.curves()

        if optimize_threshold:
            self.optimize_threshold()

        self.predict_class()
        self.compute_metrics()

    def predict_class(self):
        # self.predictions = self.model.predict(self.data.testing_features())  #labels
        self.predictions = [math.floor(p) if p < self.threshold else math.ceil(p) for p in self.scores]

    def curves(self):
        from sklearn.metrics import roc_curve, precision_recall_curve
        self.fpr, self.tpr, self.thres_roc = roc_curve(self.data.testing_labels(), self.scores, pos_label=1)
        self.prec, self.rec, self.thres_pr = precision_recall_curve(self.data.testing_labels(), self.scores, pos_label= 1)

    def compute_metrics (self):
        from sklearn.metrics import auc, f1_score, precision_score, recall_score, accuracy_score, log_loss
        self.roc_auc = auc(self.fpr, self.tpr)
        self.pr_auc = auc(self.rec, self.prec)
        self.f1 = f1_score(self.data.testing_labels(), self.predictions)
        self.p = precision_score(self.data.testing_labels(), self.predictions)
        self.r = recall_score(self.data.testing_labels(), self.predictions, pos_label= 1)
        self.specificity = recall_score(self.data.testing_labels(), self.predictions, pos_label= 0)
        self.accuracy = accuracy_score(self.data.testing_labels(), self.predictions)
        try:
            self.log_loss = log_loss(self.data.testing_labels(), self.predictions)
        except:
            self.log_loss = 999.
            Warning('Log loss not computed')

    def optimize_threshold(self):
        # With F-Score
        fscore = (2 * self.prec * self.rec) / (self.prec + self.rec)
        self.best_threshold_index_pr = np.nanargmax(fscore)
        self.threshold = self.thres_pr[self.best_threshold_index_pr]
    

    def write_report(self, extra_args: dict = {}, to_file = True):
        # save a txt file with metrics and arguments used.
        pass 

    def _metrics_report(self):
        from sklearn.metrics import classification_report
        a = [
            'ROC curve AUC: {}\n'.format(self.roc_auc),
            'Precision-recall AUC: {}\n'.format(self.pr_auc),
            f'F1-score: {self.f1}\n',
            f'Precision: {self.p}\n',
            f'Recall: {self.r}\n',
            f'Specificity: {self.specificity}\n',
            f'Accuracy: {self.accuracy}\n',
            f'Log loss: {self.log_loss}\n'
            f'Threshold: {self.threshold}\n',
            '-'*70 + '\n',
            classification_report(self.data.testing_labels(),self.predictions),
        ]
        return a

    def plot_report(self, loss_lims = (None,None), to_file = True):
        # save the plots given by the "plot" attribute to a file.

        plt.figure(figsize=self.figsize)

        for i,plot in enumerate(self.plots):
            plt.subplot(self.rows, self.cols, i+1)
            match plot:
                case "confusion matrix":
                    self.conf_matrix()
                case "roc":
                    self.roc()
                case "precision-recall":
                    self.precision_recall()
                case "loss":
                    self.loss_epochs(loss_lims)
        if to_file:
            plt.savefig(f'metrics/{self.name}.png', dpi=150, bbox_inches= 'tight')
            plt.close()
        else:
            plt.show()

    def conf_matrix(self):
        from sklearn.metrics import confusion_matrix
        conf_m = confusion_matrix(self.data.testing_labels(), self.predictions)
        df_conf_m = pd.DataFrame(conf_m, index=[0,1], columns=[0,1])
        sns.heatmap(df_conf_m, cmap=sns.color_palette('light:teal', as_cmap=True), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')

    def roc(self):
        plt.plot(self.fpr, self.tpr)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid()

    def precision_recall(self):
        a = len (self.data.testing[self.data.testing_labels()==1])/len(self.data.testing)
        plt.plot(self.rec, self.prec)
        plt.plot([0, 1], [a, a] , color='gray', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(self.rec[self.best_threshold_index_pr], self.prec[self.best_threshold_index_pr], marker='o', color='black')

    def loss_epochs(self, loss_lims = (None,None)):
        # loss during training for neural networks
        plt.plot(self.history['loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim(loss_lims)
        plt.grid()
        plt.legend()

    def plot_importances(self, importances: list, sort_importances: str|None, to_file: bool = True, permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):

        self.compute_importances(importances= importances, sort_importances= sort_importances, permutation_train_max_samples= permutation_train_max_samples, permutation_test_max_samples= permutation_test_max_samples)
        self._plot_importances(to_file= to_file)
    
    def compute_importances(self,importances: list, sort_importances: str|None, permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):

        imp = dict(feature = self.data.features)

        for i in importances:
            match i:
                case 'permutation_train':
                    perm_train = self.permutation_importance(kind= 'training', n_samples=permutation_train_max_samples)
                    imp[i] = perm_train
                case 'permutation_test':
                    perm_test = self.permutation_importance(kind = 'testing', n_samples= permutation_test_max_samples)
                    imp[i] = perm_test
                case 'gini':
                    gini = self.model.feature_importances_
                    imp[i] = gini


        # save to csv
        imp = pd.DataFrame(imp)
        imp.to_csv(f'metrics/{self.name}_importances.csv', index= False)
        if sort_importances:
            imp = imp.sort_values(by = sort_importances, ascending= False)

        self.importances = imp

    def _plot_importances(self, to_file: bool = True):
        
        imp = pd.melt(self.importances, var_name="type", value_name="importance", id_vars= 'feature')

        plt.figure(figsize= (10, len(self.data.features)/2.5))
        sns.barplot(data = imp, x = 'importance', y = 'feature', hue = 'type', orient = 'h')
        plt.grid()
        if to_file:
            plt.savefig(f'metrics/{self.name}_importances.png', dpi=150, bbox_inches= 'tight')
            plt.close()   
        else: 
            plt.show()

    def permutation_importance(self, kind: str, n: int = 3, n_samples: int|float = 1.0):

        import numpy as np
        from sklearn.utils import shuffle
        from sklearn.metrics import average_precision_score as ap

        match kind:
            case 'training':
                max_samples = self.data.training.shape[0]
                mem = self.data.training[self.data.features + ['member']][self.data.training.member == 1].values
                nmem = self.data.training[self.data.features + ['member']][self.data.training.member == 0].values
            case 'testing':
                max_samples = self.data.testing.shape[0]
                mem = self.data.testing[self.data.features + ['member']][self.data.testing.member == 1].values
                nmem = self.data.testing[self.data.features + ['member']][self.data.testing.member == 0].values

        features = self.data.features
        importances = np.zeros(len(features))
        
        if n_samples >= 1.0: n_samples = min(n_samples/max_samples, 1.)

        for i,feature in enumerate(features):

            news = []
            # permute n times
            for j in range(n):
                
                print(f'Computing {kind} importance for {feature} ({j+1}/{n})', end='\r', flush= True)

                a = mem[np.random.choice(mem.shape[0], math.floor(n_samples*mem.shape[0]), replace=False), :]
                b = nmem[np.random.choice(nmem.shape[0], math.floor(n_samples*nmem.shape[0]), replace=False), :]
                sample = np.vstack([a,b])

                shuffled = shuffle(sample[:,i], random_state= 42)
                sample[:,i] = shuffled

                scores = self.return_score(sample[:,:-1])
                news.append(ap(sample[:,-1], scores))

                print('\033[K', end= '\r')

            importances[i] = self.pr_auc - np.mean(news) 
        return importances
    
    def return_score(self, sample: pd.DataFrame):
        pass
        


    # i have never used these methods

    # def test_thresholds(self, thresholds: list|None, to_file: bool = True):
    #     # get predictions with different thresholds
    #     pred_c_thres = []
    #     for t in thresholds:
    #         pred_c_thres.append([math.floor(p[0]) if p[0] < t else math.ceil(p[0]) for p in self.scores])

    #     # plot conf matrix for each
    #     from sklearn.metrics import confusion_matrix
    #     plt.figure(figsize=(15,4))
    #     for i,(p,t) in enumerate(zip(pred_c_thres,thresholds)):
    #         plt.subplot(1, 3, i+1)
    #         plt.title(f'threshold = {t}')
    #         c = confusion_matrix(self.data.testing_labels(), p)
    #         c = pd.DataFrame(c, index=[0,1], columns=[0,1])
    #         sns.heatmap(c, cmap=sns.color_palette('light:indigo', as_cmap=True), annot=True, fmt='d')
    #         plt.xlabel('Predicted')
    #         plt.ylabel('True')
    #     if to_file:
    #         plt.savefig(f'metrics/{self.name}_thresholds.png', dpi=150, bbox_inches= 'tight')
    #         plt.close()
    #     else:
    #         plt.show()

    # def test_best_thresholds(self):
    #     from sklearn.metrics import confusion_matrix

    #     # With G-Mean
    #     gmeans = (self.tpr * (1-self.fpr))**(1/2)
    #     max_gmeans = np.argmax(gmeans) # locate the index of the largest g-mean
    #     # ROC Curve
    #     plt.figure(figsize=(10,10))
    #     plt.subplot(2, 2, 1)
    #     plt.plot(self.fpr, self.tpr)
    #     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    #     plt.plot(self.fpr[max_gmeans], self.tpr[max_gmeans], marker='o', color='black')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('FPR')
    #     plt.ylabel('TPR')
    #     plt.title(f'G-Mean = {gmeans[max_gmeans]:.3f}')
    #     plt.grid()
    #     # Conf matrix
    #     pred_gmeans = [math.floor(p[0]) if p[0] < self.thres_roc[max_gmeans] else math.ceil(p[0]) for p in self.scores]
    #     plt.subplot(2, 2, 2)
    #     plt.title(f'Threshold = {self.thres_roc[max_gmeans] :.3f}')
    #     c = confusion_matrix(self.data.testing_labels, pred_gmeans)
    #     c = pd.DataFrame(c, index=[0,1], columns=[0,1])
    #     sns.heatmap(c, cmap=sns.color_palette('light:chocolate', as_cmap=True), annot=True, fmt='d')
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')

    #     # With F-Score
    #     fscore = (2 * self.prec * self.rec) / (self.prec + self.rec)
    #     max_f = np.argmax(fscore)

    #     # Precision-recall curve
    #     plt.subplot(2, 2, 3)
    #     plt.plot(self.rec, self.prec)
    #     plt.plot(self.rec[max_f], self.prec[max_f], marker='o', color='black')
    #     a = len (self.data.testing_labels()[self.data.testing_labels() == 1])/len(self.data.testing_labels())
    #     plt.plot([0, 1], [a, a] , color='gray', linestyle='--')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title(f'F-Score = {fscore[max_f] :.3f}')
    #     plt.grid()
    #     plt.xlim([0,1])
    #     plt.ylim([0,1])
    #     # Conf matrix
    #     pred_f = [math.floor(p[0]) if p[0] < self.thres_pr[max_f] else math.ceil(p[0]) for p in self.scores]
    #     plt.subplot(2, 2, 4)
    #     c = confusion_matrix(self.data.testing_labels(), pred_f)
    #     c = pd.DataFrame(c, index=[0,1], columns=[0,1])
    #     sns.heatmap(c, cmap=sns.color_palette('light:mediumvioletred', as_cmap=True), annot=True, fmt='d')
    #     plt.title(f'Threshold = {self.thres_pr[max_f] :.3f}')
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')

    #     plt.savefig(f'metrics/{self.name}_best-thresholds_.png', dpi=150, bbox_inches= 'tight')
    #     plt.close()

