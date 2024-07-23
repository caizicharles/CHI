import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc, \
    accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score


class MetricBase():

    def __init__(self, *args, **kwargs) -> None:
        pass

    def calculate(self, probability, target):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError


class AUROC(MetricBase):

    def __init__(self, task):

        self.NAME = 'AUROC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return roc_auc_score(target, probability)

        elif self.task == 'drug_recommendation':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return roc_auc_score(target, probability, average="samples")

        elif self.task == 'los_prediction':
            #TODO
            return roc_auc_score(target, probability, multi_class="ovr", average="weighted")

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


class AUPRC(MetricBase):

    def __init__(self, task):

        self.NAME = 'AUPRC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            (precisions, recalls, _) = precision_recall_curve(target, probability)
            return auc(recalls, precisions)

        elif self.task == 'drug_recommendation':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            (precisions, recalls, _) = precision_recall_curve(target, probability)
            return auc(recalls, precisions)

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


class AP(MetricBase):

    def __init__(self, task):

        self.NAME = 'AP'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return average_precision_score(target, probability)

        elif self.task == 'drug_recommendation':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return average_precision_score(target, probability, average="samples")

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


METRICS = {'AUROC': AUROC, 'AUPRC': AUPRC, 'AP': AP, 'F1': None, 'Accuracy': None, 'Kappa': None, 'Jaccard': None}
