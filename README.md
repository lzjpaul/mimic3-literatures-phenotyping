# Mimic3-Literatures-Phenotyping
Original experimental code of the model which combines LDA with MLP
### Datasets：
* MIMICIII data (https://mimic.physionet.org/about/mimic/)
* PubMed (https://www.ncbi.nlm.nih.gov/pmc/)

### Data Preprocessing：
1. MIMICIII data prepeocessing for MLP  
    * /scripts/generate_all_event.py   
    Retrieve three kind events(diagnoses, labtests, prescriptions) from MIMICIII data. Diagnoses for ICD9_CODE; LabTests for ITEMID; Prescription for FORMULARY_DRUG_CD. Each code must appear several times in datasets.
    * /scripts/create_diagnoses_dict.py  
    Generate a dict for all medical events, and retrieve a formal description for each event inoder to do Named Entity Recognization in medical articles.
    * /scripts/generate_instance.py  
    Retrieve the instances from event sequences. Every instance includes diagnoses of one bill and all events in 90 days before the bill. The predictive task is to predict the diagnoses in the bill.
     
2. Medical articles preprocessing for trianing LDA   
    * /scripts/select_relate_literature.py  
      /scripts/select_generate_new_docs.py   
    Find the events in medical articles using events' descriptions, replace the article contend by several events.  
    
### Models：
* /baseline_method/pytorch_MLP.py  
    Combine the LDA with MLP by add the inference result as penalty term to the objective function of MLP.
    