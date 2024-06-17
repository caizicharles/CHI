import csv
from pyhealth.datasets import MIMIC4Dataset


def process_dataset(dataset_name):

    if dataset_name == "mimic-iv":
        print(1)
        dataset = MIMIC4Dataset(
        root="C:/Users/caizi/Desktop/mimiciv/2.2/hosp/", 
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],      
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC",
            "ICD10CM": "CCSCM",
            "ICD10PROC": "CCSPROC",
            },
        dev=False
        )

    return dataset
