import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score


class MetricBase():

    def __init__(self, *args, **kwargs) -> None:
        pass

    def calculate(self, prediction, target):
        raise NotImplementedError

    def log(self):
        pass


class AUPRC(MetricBase):

    def __init__(self, task):

        self.NAME = 'AUPRC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return  average_precision_score(target, probability)
        
        elif self.task == 'drug_recommendation':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            prediction = (probability >= 0.5).astype(int)
            return average_precision_score(target, prediction, average="samples")
        
        elif self.task == 'los_prediction':
            return 0
        
    def log(self, score, logger, writer=None, global_iter=None, name_prefix=''):
        logger.info(f'AUPRC: {score:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', score, global_iter)

def init_metrics(args):
    
    METRICS = {
        'AUROC': None,
        'AUPRC': AUPRC(args.task),
        'F1': None,
        'Accuracy': None,
        'Kappa': None,
        'Jaccard': None
    }

    return METRICS
