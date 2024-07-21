import sqlite3
import tqdm
from database_guideline import *

from utils import delete, delete_empty
from transformers import AutoTokenizer, BioGptModel, AutoModel, BioGptTokenizer, BioGptForCausalLM
from transformers.activations import ACT2FN
import os
import torch
import torch.nn as nn
import faiss
import tqdm
import argparse
    # Connect to DB and create a cursor
conn = sqlite3.connect('../physionet.org/files/mimiciii/1.4/mimic3.db')
config = {
    "pooler_hidden_size": 768,
    "batch_size": 1,
}
config = argparse.Namespace(**config)
cursor = conn.cursor()

# Query to retrieve all subject IDs
query = "SELECT subject_id, hadm_id FROM noteevents"

# Execute the query
cursor.execute(query)

# Fetch all subject IDs
ids = cursor.fetchall()
#subject_ids = [id[0] for id in subject_ids]

#print(ids[0])
#print(len(ids))

# Query to retrieve all admission IDs for the given subject ID
#query = "SELECT hadm_id FROM admissions WHERE subject_id = ?"
#admission_ids = []
#for subject_id in subject_ids:
# Execute the query
#    cursor.execute(query, (subject_id,))
#    tmp = cursor.fetchall()
#    admission_ids.append(tmp)
#print(admission_ids[0])


# add index for each table
def add_index():
    try:
        cursor.execute("CREATE INDEX idx_admissions ON admissions(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_callout ON callout(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_caregivers ON caregivers(CGID)")
        cursor.execute("CREATE INDEX idx_chartevents ON chartevents(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_cptevents ON cptevents(SUBJECT_ID, HADM_ID)")
# cursor.execute("CREATE INDEX idx_d_cpt ON d_cpt(CPT_CD)")
        cursor.execute("CREATE INDEX idx_d_icd_diagnoses ON d_icd_diagnoses(ICD9_CODE)")
        cursor.execute("CREATE INDEX idx_d_icd_procedures ON d_icd_procedures(ICD9_CODE)")
        cursor.execute("CREATE INDEX idx_d_items ON d_items(ITEMID)")
        cursor.execute("CREATE INDEX idx_d_labitems ON d_labitems(ITEMID)")
        cursor.execute("CREATE INDEX idx_datetimeevents ON datetimeevents(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_diagnoses_icd ON diagnoses_icd(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_drgcodes ON drgcodes(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_icustays ON icustays(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_inputevents_cv ON inputevents_cv(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_inputevents_mv ON inputevents_mv(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_labevents ON labevents(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_microbiologyevents ON microbiologyevents(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_noteevents ON noteevents(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_outputevents ON outputevents(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_patients ON patients(SUBJECT_ID)")
        cursor.execute("CREATE INDEX idx_prescriptions ON prescriptions(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_procedureevents_mv ON procedureevents_mv(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
        cursor.execute("CREATE INDEX idx_procedures_icd ON procedures_icd(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_services ON services(SUBJECT_ID, HADM_ID)")
        cursor.execute("CREATE INDEX idx_transfers ON transfers(SUBJECT_ID, HADM_ID, ICUSTAY_ID)")
    except:
        print("Index already exists")

class NonParamPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        return context_token

    @property
    def output_dim(self):
        return self.config.hidden_size


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size



def save_data():
    data = {}
    for subject_id, admission_id in tqdm.tqdm(ids):

        icu_query = """
            SELECT ICUSTAY_ID
            FROM icustays
            WHERE SUBJECT_ID = ?
            AND HADM_ID = ?

        """    
        admission = ADMISSION().get(subject_id, admission_id, conn)
        callout = CALLOUT().get(subject_id, admission_id, conn)
        cursor.execute(icu_query, (subject_id, admission_id))
        icustay_ids = [t[0] for t in cursor.fetchall()]
        chartevents = {icustay_id: CHARTEVENTS().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        for icustay_id in chartevents:
            if "ITEMID" in chartevents:
                chartevents[icustay_id]['ITEMID'] = D_ITEMS().get(chartevents[icustay_id]['ITEMID'], conn)
        for icustay_id in chartevents:
            if "CGID" in chartevents[icustay_id]:
                #print("chartevents[icustay_id]: ", type(chartevents[icustay_id]))
                chartevents[icustay_id]["CAREGIVERS"] = CAREGIVERS().get(chartevents[icustay_id]["CGID"], conn)
        cptevents = CPTEVENTS().get(subject_id, admission_id, conn)
        if "CPT_CD" in cptevents:
            cptevents["D_CPT"] = D_CPT().get(cptevents["CPT_CD"], conn)
        datetimeevents = {icustay_id: DATETIMEEVENTS().get(subject_id, admission_id, icustay_id, conn)}
        for icustay_id in datetimeevents:
            if "ITEMID" in datetimeevents[icustay_id]:
                datetimeevents[icustay_id]["D_ITEMS"] = D_ITEMS().get(datetimeevents[icustay_id]['ITEMID'], conn)
            if "CGID" in datetimeevents:
                datetimeevents[icustay_id]["CAREGIVERS"] = CAREGIVERS().GET(datetimeevents[icustay_id]["CGID"], conn)
        diagnoses_icd = DIAGNOSES_ICD().get(subject_id, admission_id, conn)
        drgcodes = DRGCODES().get(subject_id, admission_id, conn)
        icustays = {icustay_id: ICUSTAYS().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        inputevents_cv = {icustay_id: INPUTEVENTS_CV().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        inputevents_mv = {icustay_id: INPUTEVENTS_MV().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        labevents = LABEVENTS().get(subject_id, admission_id, conn)
        microbiologyevents = MICROBIOLOGYEVENTS().get(subject_id, admission_id, conn)
        noteevents = NOTEEVENTS().get(subject_id, admission_id, conn)
        outputevents = {icustay_id: OUTPUTEVENTS().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        patients = PATIENTS().get(subject_id, conn)
        prescriptions = {icustay_id: PRESCRIPTIONS().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        procedures_icd = PROCEDURES_ICD().get(subject_id, admission_id, conn)
        procedureevents_mv = {icustay_id: PROCEDUREEVENTS_MV().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}
        services = SERVICES().get(subject_id, admission_id, conn)
        transfers = {icustay_id: TRANSFERS().get(subject_id, admission_id, icustay_id, conn) for icustay_id in icustay_ids}



        #print("admission: ", admission)
        #print("callout: ", callout)
        #print("icustayids: ", icustay_ids)
        #print("chartevents: ", chartevents)
        #print("cptevents: ", cptevents)
        #print("datetimeevents: ", datetimeevents)
        #print("diagnoses_icd: ", diagnoses_icd)
        #print("drgcodes: ", drgcodes)
        #print("icustays: ", icustays)
        #print("inputevents_cv: ", inputevents_cv)
        #print("inputevents_mv: ", inputevents_mv)
        #print("labevents: ", labevents)
        #print("microbiologyevents: ", microbiologyevents)
        #print("noteevents: ", noteevents)
        #print("outputevents: ", outputevents)
        #print("patients: ", patients)
        #print("prescriptions: ", prescriptions)
        #print("procedures_icd: ", procedures_icd)
        #print("procedureevents_mv: ", procedureevents_mv)
        #print("services: ", services)
        #print("transfers: ", transfers)
        #print(type(admission))
        context = {
            "admission": admission,
            "callout": callout,
            "chartevents": chartevents,
            "cptevents": cptevents,
            "datetimeevents": datetimeevents,
            "diagnoses_icd": diagnoses_icd,
            "drgcodes": drgcodes,
            "icustays": icustays,
            "inputevents_cv": inputevents_cv,
            "inputevents_mv": inputevents_mv,
            "labevents": labevents,
            "microbiologyevents": microbiologyevents,
            "noteevents": noteevents,
            "outputevents": outputevents,
            "patients": patients,
            "prescriptions": prescriptions,
            "procedures_icd": procedures_icd,
            "procedureevents_mv": procedureevents_mv,
            "services": services,
            "transfers": transfers
        }
        #delete(context)
        #print("context: ", context)
        #break
        del context["prescriptions"]
        del context['noteevents']
        delete(context)
        delete_empty(context)
        print("context: ", context)
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        context = json.dumps(context)
        inputs = tokenizer(context, return_tensors="pt")
        print("inputs.shape: ", inputs.input_ids.shape)
        data[f"({subject_id}, {admission_id})"] = context
        break
    data = json.dumps(data)
    with open("data.txt", "w") as my_file:
        my_file.write(data)
        
    print("done")


def get_data():
    with open("data.txt", "r") as my_file:
        data = my_file.read()
        data = json.loads(data)
    res = {}
    for key in data:
        res[tuple(key)] = data[key]
    return res

if __name__ == "__main__":
    #save_data
    #save_data()
    data = get_data()

    X, Y = [], []

    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptModel.from_pretrained("microsoft/biogpt").to("cuda")

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(**inputs)

    # encoding = outputs.last_hidden_state

    for key in tqdm.tqdm(data):
        context = data[key]
        Y.append(context["prescriptions"])
        del context["prescriptions"]
        del context['noteevents']
        X.append(json.dumps(context))

    #print("X[0]: ", X[0])
    #print("X[1]: ", X[1])
        #break
    try:
        encodings = torch.load("cache/encodings.pt").cpu().numpy()
        #raise Exception
    except Exception as e:
        #print(e)
        encodings = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(X), config.batch_size)):
                print("X[0]", X[0])
                inputs = tokenizer(X[i: i+config.batch_size], return_tensors="pt").to("cuda")

                #print("type of inputs: ", type(inputs))
                #print(inputs[:2])
                print("inputs.shape: ", inputs.input_ids.shape)
                break
                outputs = model(**(inputs))
                #print(outputs.last_hidden_state.shape)
                encoding = outputs.last_hidden_state
                #print(torch.eq(encoding[0], encoding[1]))
                encoding = torch.mean(encoding, dim=1)
                #encoding = NonParamPooler(config)(encoding)
                #print(encoding)
                #if i == 2*config.batch_size:

                #break
                encodings.append(encoding)
                #break
        encodings = torch.cat(encodings, dim=0)
        torch.save(encodings, "cache/encodings.pt")
        encodings = encodings.detach().cpu().numpy()
    #torch.save(X, "X.pt")


    print("encodings.shape: ", encodings.shape)
    # Dimension of vectors
    d = encodings.shape[1]
    # GPU config
    
    gpu_ids = "0"  # can be e.g. "3,4" for multiple GPUs 
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    # Setup
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(encodings)
    print("encodings[0]", encodings[0])
    print("encodings[1]", encodings[1])
    print("encodings[100]", encodings[100])
    print("encodings[200]", encodings[200])
    #torch.save(X, "X.pt")
    # Create an index
    #index = faiss.IndexFlatL2(d)

    # Add vectors to the index
    #index.add(encodings)

    # Search for the nearest neighbors of a vector
    train_X = encodings[:10]
    print(train_X)
    D, I = gpu_index.search(train_X, 2)
    print("D: ", D)
    print("I: ", I)
