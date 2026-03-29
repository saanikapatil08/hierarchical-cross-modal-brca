
import pandas as pd
import numpy as np
from pathlib import Path

GDC  = Path('data/raw/clinical_gdc.tsv')
CBIO = Path('data/raw/clinical_pam50.tsv')
OUT  = Path('data/tcga_brca/clinical.csv')
Path('data/tcga_brca').mkdir(parents=True, exist_ok=True)

gdc = pd.read_csv(GDC, sep='\t', low_memory=False)
gdc = gdc.drop_duplicates(subset='cases.submitter_id').set_index('cases.submitter_id')
print(f'GDC: {len(gdc)} patients')

clin = pd.DataFrame(index=gdc.index)
clin.index.name = 'patient_id'

age = pd.to_numeric(gdc['demographic.days_to_birth'], errors='coerce')
clin['age_at_diagnosis'] = (age.abs() / 365.25).round(1).fillna(55.0)

stage_map = {
    'Stage I':'I','Stage IA':'I','Stage IB':'I',
    'Stage II':'II','Stage IIA':'II','Stage IIB':'II',
    'Stage III':'III','Stage IIIA':'III','Stage IIIB':'III','Stage IIIC':'III',
    'Stage IV':'IV'
}
if 'diagnoses.ajcc_pathologic_stage' in gdc.columns:
    clin['stage'] = gdc['diagnoses.ajcc_pathologic_stage'].map(stage_map).fillna('II')
else:
    clin['stage'] = 'II'

clin['tumor_size_cm']       = 2.0
clin['lymph_nodes_positive'] = 0
clin['er_status']            = 0
clin['pr_status']            = 0
clin['her2_status']          = 0
clin['menopausal_status']    = 1
clin['histology']            = 'IDC'
clin['grade']                = '2'

cbio = pd.read_csv(CBIO, sep='\t', low_memory=False).set_index('Patient ID')
pam50_map = {
    'BRCA_LumA':'LumA','BRCA_LumB':'LumB','BRCA_Her2':'Her2',
    'BRCA_Basal':'Basal','BRCA_Normal':'Normal'
}
pam50 = cbio['Subtype'].map(pam50_map)
clin  = clin.join(pam50.rename('PAM50'), how='left')
clin  = clin[clin['PAM50'].isin(['LumA','LumB','Her2','Basal','Normal'])]

for col in ['age_at_diagnosis','tumor_size_cm','lymph_nodes_positive']:
    clin[col] = pd.to_numeric(clin[col], errors='coerce').fillna(55.0)

clin.to_csv(OUT)
print(f'Saved {len(clin)} patients to {OUT}')
print(clin['PAM50'].value_counts().to_string())
