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

    def main(self, optimize_threshold: bool = True, extra_args: dict = {}, loss_lims: tuple = (None, None)):
    # thresholds: list|None = None, best_thresholds: bool = False
        
        self.predict(optimize_threshold)
        self.write_report(extra_args)
        self.plot_report(loss_lims= loss_lims)

        # # test different thresholds
        # if thresholds:
        #     self.test_thresholds(thresholds)

        # if best_thresholds:
        #     self.test_best_thresholds()

    def predict(self, optimize_threshold: bool = True):
        self.predict_score()
        self.curves()

        if optimize_threshold:
            self.optimize_threshold()

        self.predict_class()
        self.compute_metrics()
        
    
    def predict_score(self):
        scores = self.model.predict_proba(self.data.testing_features())  #probabilities
        self.scores = scores[:,1]

    def predict_class(self):
        # self.predictions = self.model.predict(self.data.testing_features())  #labels
        self.predictions = [math.floor(p) if p < self.threshold else math.ceil(p) for p in self.scores]

    def curves(self):
        from sklearn.metrics import roc_curve, precision_recall_curve
        self.fpr, self.tpr, self.thres_roc = roc_curve(self.data.testing_labels(), self.scores, pos_label=1)
        self.prec, self.rec, self.thres_pr = precision_recall_curve(self.data.testing_labels(), self.scores, pos_label= 1)

    def compute_metrics (self):
        from sklearn.metrics import auc, f1_score, precision_score, recall_score, accuracy_score
        self.roc_auc = auc(self.fpr, self.tpr)
        self.pr_auc = auc(self.rec, self.prec)
        self.f1 = f1_score(self.data.testing_labels(), self.predictions)
        self.p = precision_score(self.data.testing_labels(), self.predictions)
        self.r = recall_score(self.data.testing_labels(), self.predictions, pos_label= 1)
        self.specificity = recall_score(self.data.testing_labels(), self.predictions, pos_label= 0)
        self.accuracy = accuracy_score(self.data.testing_labels(), self.predictions)

    def optimize_threshold(self):
        from sklearn.metrics import confusion_matrix

        # With F-Score
        fscore = (2 * self.prec * self.rec) / (self.prec + self.rec)
        self.best_threshold_index_pr = np.argmax(fscore)
        self.threshold = self.thres_pr[self.best_threshold_index_pr]
    

    def write_report(self, extra_args: dict = {}, to_file = True):
        # save a txt file with metrics and arguments used.
        pass 

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

