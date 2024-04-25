from ModelTester import ModelTester
import cudf

class SVCModelTester(ModelTester):

    plots = ["confusion matrix"]
    rows: int = 1
    cols: int = 1
    figsize: tuple = (5,4)

    def main(self, extra_args: dict = {}):
        self.predict()
        self.write_report(extra_args)
        self.plot_report()
    
    def predict(self):
        self.predict_class()
        self.compute_metrics()

    def predict_class(self):
        predictions = self.model.predict(cudf.from_pandas(self.data.testing_features()))
        self.predictions = predictions.to_numpy()

    def compute_metrics (self):
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        self.f1 = f1_score(self.data.testing_labels(), self.predictions)
        self.p = precision_score(self.data.testing_labels(), self.predictions)
        self.r = recall_score(self.data.testing_labels(), self.predictions, pos_label= 1)
        self.specificity = recall_score(self.data.testing_labels(), self.predictions, pos_label= 0)
        self.accuracy = accuracy_score(self.data.testing_labels(), self.predictions)

    def write_report(self, extra_args: dict = {}):

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

    def _metrics_report(self):
        from sklearn.metrics import classification_report
        a = [
            f'F1-score: {self.f1}\n',
            f'Precision: {self.p}\n',
            f'Recall: {self.r}\n',
            f'Specificity: {self.specificity}\n',
            f'Accuracy: {self.accuracy}\n',
            '-'*70 + '\n',
            classification_report(self.data.testing_labels(),self.predictions),
        ]
        return a


    def curves(self):
        raise NotImplementedError
    def optimize_threshold(self):
        raise NotImplementedError
    def _plot_importances(self):
        raise NotImplementedError
    def plot_importances(self):
        raise NotImplementedError
    def compute_importances(self):
        raise NotImplementedError
    def permutation_importance(self):
        raise NotImplementedError