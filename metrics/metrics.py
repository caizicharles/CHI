import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, jaccard_score, cohen_kappa_score


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
            return roc_auc_score(target, probability, average="samples")

        elif self.task == 'los_prediction':
            return roc_auc_score(target, probability, multi_class="ovr", average="weighted")

        elif self.task == 'pretrain':
            return roc_auc_score(target, probability, average="samples")

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
            return average_precision_score(target, probability)

        elif self.task == 'drug_recommendation':
            return average_precision_score(target, probability, average="samples")

        elif self.task == 'los_prediction':
            return 0

        elif self.task == 'pretrain':
            return average_precision_score(target, probability, average="samples")

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


class Kappa(MetricBase):

    def __init__(self, task):

        self.NAME = 'Kappa'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            probability = (probability >= 0.5).astype(int)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, probability)

        elif self.task == 'drug_recommendation':
            return

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, pred)

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


class Accuracy(MetricBase):

    def __init__(self, task):

        self.NAME = 'Accuracy'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            pred = (probability >= 0.5).astype(int)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

        elif self.task == 'drug_recommendation':
            probability = probability.flatten()
            target = target.flatten()
            pred = (probability >= 0.5).astype(int)
            return accuracy_score(target, pred)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

        elif self.task == 'pretrain':
            probability = probability.flatten()
            target = target.flatten()
            pred = (probability >= 0.5).astype(int)
            return accuracy_score(target, pred)

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


class F1(MetricBase):

    def __init__(self, task):

        self.NAME = 'F1'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            probability = (probability >= 0.5).astype(int)
            return f1_score(target, probability, average="macro", zero_division=1)

        elif self.task == 'drug_recommendation':
            pred = (probability >= 0.5).astype(int)
            return f1_score(target, pred, average="samples", zero_division=1)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return f1_score(target, pred, average="weighted")

        elif self.task == 'pretrain':
            pred = (probability >= 0.5).astype(int)
            return f1_score(target, pred, average="samples", zero_division=1)

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


class Jaccard(MetricBase):

    def __init__(self, task):

        self.NAME = 'Jaccard'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            probability = (probability >= 0.5).astype(int)
            return jaccard_score(target, pred, average="macro", zero_division=1)

        elif self.task == 'drug_recommendation':
            pred = (probability >= 0.5).astype(int)
            return jaccard_score(target, pred, average="samples", zero_division=1)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, pred)

        elif self.task == 'pretrain':
            pred = (probability >= 0.5).astype(int)
            return jaccard_score(target, pred, average="samples", zero_division=1)

    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'{self.NAME}: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)


METRICS = {'AUROC': AUROC, 'AUPRC': AUPRC, 'F1': F1, 'Accuracy': Accuracy, 'Kappa': Kappa, 'Jaccard': Jaccard}
