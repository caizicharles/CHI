def run():
    diagnoses_list = []
    procedures_list = []
    prescriptions_list = []
    visit_map = {}

    for patient in filtered_patients.values():
        for visit in patient:
            x = []
            diag = visit.get_code_list('DIAGNOSES_ICD')
            proc = visit.get_code_list('PROCEDURES_ICD')
            pres = visit.get_code_list('PRESCRIPTIONS')

            for idx, d in enumerate(diag):
                diag[idx] = diagnoses_code_to_name[d].upper()
            for idx, pro in enumerate(proc):
                proc[idx] = procedures_code_to_name[pro].upper()
            for idx, pre in enumerate(pres):
                pres[idx] = prescriptions_code_to_name[pre].upper()

            x.extend(diag)
            x.extend(proc)
            x.extend(pres)

            visit_map[visit.visit_id] = x

    save_with_pickle(visit_map, processed_data_path, 'mimiciii_visit_id_to_nodes.pickle')
    exit()
    pass


if __name__ == '__main__':
    args = get_args()
    run(args=args)
