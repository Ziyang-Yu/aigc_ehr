import json

schema = {
  "title": "MIMIC-III Data",
  "description": "Schema for MIMIC-III data including demography, lab tests, diagnosis, and medication.",
  "type": "object",
  "properties": {
    "Demography": {
      "type": "object",
      "properties": {
        "subject_id": {
          "type": "integer",
          "description": "Unique identifier for the patient."
        },
        "gender": {
          "type": "string",
          "enum": ["M", "F"],
          "description": "Gender of the patient."
        },
        "dob": {
          "type": "string",
          "format": "date",
          "description": "Date of birth of the patient."
        },
        "dod": {
          "type": "string",
          "format": "date",
          "description": "Date of death of the patient, if applicable."
        },
        "ethnicity": {
          "type": "string",
          "description": "Ethnicity of the patient."
        },
        "marital_status": {
          "type": "string",
          "description": "Marital status of the patient."
        },
        "insurance": {
          "type": "string",
          "description": "Insurance type of the patient."
        },
        "language": {
          "type": "string",
          "description": "Primary language of the patient."
        }
      },
    #   "required": ["subject_id", "gender", "dob", "ethnicity", "insurance"]
    },
    "LabTest": {
      "type": "object",
      "properties": {
        "subject_id": {
          "type": "integer",
          "description": "Unique identifier for the patient."
        },
        "itemid": {
          "type": "integer",
          "description": "Identifier for the lab test item."
        },
        "charttime": {
          "type": "string",
          "format": "date-time",
          "description": "Time when the lab test was charted."
        },
        "value": {
          "type": "string",
          "description": "Value of the lab test result."
        },
        "valuenum": {
          "type": "number",
          "description": "Numeric value of the lab test result, if applicable."
        },
        "valueuom": {
          "type": "string",
          "description": "Unit of measurement for the lab test result."
        },
        "flag": {
          "type": "string",
          "description": "Flag indicating if the lab test result is abnormal."
        }
      },
    #   "required": ["subject_id", "itemid", "charttime", "value"]
    },
    "Diagnosis": {
      "type": "object",
      "properties": {
        "subject_id": {
          "type": "integer",
          "description": "Unique identifier for the patient."
        },
        "icd9_code": {
          "type": "string",
          "description": "ICD-9 code for the diagnosis."
        },
        "seq_num": {
          "type": "integer",
          "description": "Sequence number of the diagnosis."
        },
        "short_title": {
          "type": "string",
          "description": "Short title of the diagnosis."
        },
        "long_title": {
          "type": "string",
          "description": "Long title of the diagnosis."
        }
      },
      "required": ["subject_id", "icd9_code", "seq_num"]
    },
    "Medication": {
      "type": "object",
      "properties": {
        "subject_id": {
          "type": "integer",
          "description": "Unique identifier for the patient."
        },
        "startdate": {
          "type": "string",
          "format": "date",
          "description": "Start date of the medication."
        },
        "enddate": {
          "type": "string",
          "format": "date",
          "description": "End date of the medication."
        },
        "drug_type": {
          "type": "string",
          "description": "Type of drug."
        },
        "drug": {
          "type": "string",
          "description": "Name of the drug."
        },
        "dose_val_rx": {
          "type": "number",
          "description": "Dose value prescribed."
        },
        "dose_unit_rx": {
          "type": "string",
          "description": "Unit of dose prescribed."
        },
        "route": {
          "type": "string",
          "description": "Route of administration."
        }
      },
      "required": ["subject_id", "startdate", "drug"]
    }
  },
  "required": ["Demography", "LabTest", "Diagnosis", "Medication"]
}