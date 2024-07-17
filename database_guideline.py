from pydantic import BaseModel, Field
from datetime import datetime

import sqlite3
import json

AUTOPOPULATED = "Autopopulated"

from typing import List
#import psycopg0

class ADMISSION(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ADMITTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    DISCHTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    DEATHTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ADMISSION_TYPE: str = AUTOPOPULATED  # VARCHAR(50)
    ADMISSION_LOCATION: str = AUTOPOPULATED  # VARCHAR(50)
    DISCHARGE_LOCATION: str = AUTOPOPULATED  # VARCHAR(50)
    INSURANCE: str = AUTOPOPULATED  # VARCHAR(255)
    LANGUAGE: str = AUTOPOPULATED  # VARCHAR(10)
    RELIGION: str = AUTOPOPULATED  # VARCHAR(50)
    MARITAL_STATUS: str = AUTOPOPULATED  # VARCHAR(50)
    ETHNICITY: str = AUTOPOPULATED  # VARCHAR(200)
    EDREGTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    EDOUTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    DIAGNOSIS: str = AUTOPOPULATED  # VARCHAR(300)
    HOSPITAL_EXPIRE_FLAG: int = AUTOPOPULATED  # TINYINT
    HAS_CHARTEVENTS_DATA: int = AUTOPOPULATED  # TINYINT
    
    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME,
               ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE,
               LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, EDREGTIME, EDOUTTIME,
               DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_CHARTEVENTS_DATA
        FROM admissions
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ADMITTIME, self.DISCHTIME, \
            self.DEATHTIME, self.ADMISSION_TYPE, self.ADMISSION_LOCATION, \
            self.DISCHARGE_LOCATION, self.INSURANCE, self.LANGUAGE, self.RELIGION, \
            self.MARITAL_STATUS, self.ETHNICITY, self.EDREGTIME, self.EDOUTTIME, \
            self.DIAGNOSIS, self.HOSPITAL_EXPIRE_FLAG, self.HAS_CHARTEVENTS_DATA = row

            # Convert string timestamps to datetime objects if needed
            self.ADMITTIME = datetime.strptime(self.ADMITTIME, '%Y-%m-%d %H:%M:%S') if self.ADMITTIME else None
            self.DISCHTIME = datetime.strptime(self.DISCHTIME, '%Y-%m-%d %H:%M:%S') if self.DISCHTIME else None
            self.DEATHTIME = datetime.strptime(self.DEATHTIME, '%Y-%m-%d %H:%M:%S') if self.DEATHTIME else None
            self.EDREGTIME = datetime.strptime(self.EDREGTIME, '%Y-%m-%d %H:%M:%S') if self.EDREGTIME else None
            self.EDOUTTIME = datetime.strptime(self.EDOUTTIME, '%Y-%m-%d %H:%M:%S') if self.EDOUTTIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ADMITTIME": self.ADMITTIME.isoformat() if self.ADMITTIME else None,
                "DISCHTIME": self.DISCHTIME.isoformat() if self.DISCHTIME else None,
                "DEATHTIME": self.DEATHTIME.isoformat() if self.DEATHTIME else None,
                "ADMISSION_TYPE": self.ADMISSION_TYPE,
                "ADMISSION_LOCATION": self.ADMISSION_LOCATION,
                "DISCHARGE_LOCATION": self.DISCHARGE_LOCATION,
                "INSURANCE": self.INSURANCE,
                "LANGUAGE": self.LANGUAGE,
                "RELIGION": self.RELIGION,
                "MARITAL_STATUS": self.MARITAL_STATUS,
                "ETHNICITY": self.ETHNICITY,
                "EDREGTIME": self.EDREGTIME.isoformat() if self.EDREGTIME else None,
                "EDOUTTIME": self.EDOUTTIME.isoformat() if self.EDOUTTIME else None,
                "DIAGNOSIS": self.DIAGNOSIS,
                "HOSPITAL_EXPIRE_FLAG": self.HOSPITAL_EXPIRE_FLAG,
                "HAS_CHARTEVENTS_DATA": self.HAS_CHARTEVENTS_DATA,
            }
        else:
            result = {}

        cursor.close()
        return result

class CALLOUT(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    SUBMIT_WARDID: int = AUTOPOPULATED
    SUBMIT_CAREUNIT: str = AUTOPOPULATED  # VARCHAR(15)
    CURR_WARDID: int = AUTOPOPULATED
    CURR_CAREUNIT: str = AUTOPOPULATED  # VARCHAR(15)
    CALLOUT_WARDID: int = AUTOPOPULATED
    CALLOUT_SERVICE: str = AUTOPOPULATED  # VARCHAR(10)
    REQUEST_TELE: int = AUTOPOPULATED  # SMALLINT
    REQUEST_RESP: int = AUTOPOPULATED  # SMALLINT
    REQUEST_CDIFF: int = AUTOPOPULATED  # SMALLINT
    REQUEST_MRSA: int = AUTOPOPULATED  # SMALLINT
    REQUEST_VRE: int = AUTOPOPULATED  # SMALLINT
    CALLOUT_STATUS: str = AUTOPOPULATED  # VARCHAR(20)
    CALLOUT_OUTCOME: str = AUTOPOPULATED  # VARCHAR(20)
    DISCHARGE_WARDID: int = AUTOPOPULATED
    ACKNOWLEDGE_STATUS: str = AUTOPOPULATED  # VARCHAR(20)
    CREATETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    UPDATETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ACKNOWLEDGETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    OUTCOMETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    FIRSTRESERVATIONTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CURRENTRESERVATIONTIME: str = AUTOPOPULATED  # TIMESTAMP(0)

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, SUBMIT_WARDID, SUBMIT_CAREUNIT, CURR_WARDID, 
               CURR_CAREUNIT, CALLOUT_WARDID, CALLOUT_SERVICE, REQUEST_TELE, REQUEST_RESP, 
               REQUEST_CDIFF, REQUEST_MRSA, REQUEST_VRE, CALLOUT_STATUS, CALLOUT_OUTCOME, 
               DISCHARGE_WARDID, ACKNOWLEDGE_STATUS, CREATETIME, UPDATETIME, ACKNOWLEDGETIME, 
               OUTCOMETIME, FIRSTRESERVATIONTIME, CURRENTRESERVATIONTIME
        FROM callout
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.SUBMIT_WARDID, self.SUBMIT_CAREUNIT, \
            self.CURR_WARDID, self.CURR_CAREUNIT, self.CALLOUT_WARDID, self.CALLOUT_SERVICE, \
            self.REQUEST_TELE, self.REQUEST_RESP, self.REQUEST_CDIFF, self.REQUEST_MRSA, \
            self.REQUEST_VRE, self.CALLOUT_STATUS, self.CALLOUT_OUTCOME, self.DISCHARGE_WARDID, \
            self.ACKNOWLEDGE_STATUS, self.CREATETIME, self.UPDATETIME, self.ACKNOWLEDGETIME, \
            self.OUTCOMETIME, self.FIRSTRESERVATIONTIME, self.CURRENTRESERVATIONTIME = row

            # Convert string timestamps to datetime objects if needed
            self.CREATETIME = datetime.strptime(self.CREATETIME, '%Y-%m-%d %H:%M:%S') if self.CREATETIME else None
            self.UPDATETIME = datetime.strptime(self.UPDATETIME, '%Y-%m-%d %H:%M:%S') if self.UPDATETIME else None
            self.ACKNOWLEDGETIME = datetime.strptime(self.ACKNOWLEDGETIME, '%Y-%m-%d %H:%M:%S') if self.ACKNOWLEDGETIME else None
            self.OUTCOMETIME = datetime.strptime(self.OUTCOMETIME, '%Y-%m-%d %H:%M:%S') if self.OUTCOMETIME else None
            self.FIRSTRESERVATIONTIME = datetime.strptime(self.FIRSTRESERVATIONTIME, '%Y-%m-%d %H:%M:%S') if self.FIRSTRESERVATIONTIME else None
            self.CURRENTRESERVATIONTIME = datetime.strptime(self.CURRENTRESERVATIONTIME, '%Y-%m-%d %H:%M:%S') if self.CURRENTRESERVATIONTIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "SUBMIT_WARDID": self.SUBMIT_WARDID,
                "SUBMIT_CAREUNIT": self.SUBMIT_CAREUNIT,
                "CURR_WARDID": self.CURR_WARDID,
                "CURR_CAREUNIT": self.CURR_CAREUNIT,
                "CALLOUT_WARDID": self.CALLOUT_WARDID,
                "CALLOUT_SERVICE": self.CALLOUT_SERVICE,
                "REQUEST_TELE": self.REQUEST_TELE,
                "REQUEST_RESP": self.REQUEST_RESP,
                "REQUEST_CDIFF": self.REQUEST_CDIFF,
                "REQUEST_MRSA": self.REQUEST_MRSA,
                "REQUEST_VRE": self.REQUEST_VRE,
                "CALLOUT_STATUS": self.CALLOUT_STATUS,
                "CALLOUT_OUTCOME": self.CALLOUT_OUTCOME,
                "DISCHARGE_WARDID": self.DISCHARGE_WARDID,
                "ACKNOWLEDGE_STATUS": self.ACKNOWLEDGE_STATUS,
                "CREATETIME": self.CREATETIME.isoformat() if self.CREATETIME else None,
                "UPDATETIME": self.UPDATETIME.isoformat() if self.UPDATETIME else None,
                "ACKNOWLEDGETIME": self.ACKNOWLEDGETIME.isoformat() if self.ACKNOWLEDGETIME else None,
                "OUTCOMETIME": self.OUTCOMETIME.isoformat() if self.OUTCOMETIME else None,
                "FIRSTRESERVATIONTIME": self.FIRSTRESERVATIONTIME.isoformat() if self.FIRSTRESERVATIONTIME else None,
                "CURRENTRESERVATIONTIME": self.CURRENTRESERVATIONTIME.isoformat() if self.CURRENTRESERVATIONTIME else None
            }
        else:
            result = {}

        cursor.close()
        return result


class CAREGIVERS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    CGID: int = AUTOPOPULATED
    LABEL: str = AUTOPOPULATED  # VARCHAR(15)
    DESCRIPTION: str = AUTOPOPULATED  # VARCHAR(30)

    def get(self, cg_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, CGID, LABEL, DESCRIPTION
        FROM caregivers
        WHERE CGID = ?
        """
        cursor.execute(query, (cg_id,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.CGID, self.LABEL, self.DESCRIPTION = row

            result = {
                "ROW_ID": self.ROW_ID,
                "CGID": self.CGID,
                "LABEL": self.LABEL,
                "DESCRIPTION": self.DESCRIPTION
            }
        else:
            result = {}

        cursor.close()
        return result

class CHARTEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED  # NUMBER(7,0)
    HADM_ID: int = AUTOPOPULATED  # NUMBER(7,0)
    ICUSTAY_ID: int = AUTOPOPULATED  # NUMBER(7,0)
    ITEMID: int = AUTOPOPULATED  # NUMBER(7,0)
    CHARTTIME: str = AUTOPOPULATED  # DATE
    STORETIME: str = AUTOPOPULATED  # DATE
    CGID: int = AUTOPOPULATED  # NUMBER(7,0)
    VALUE: str = AUTOPOPULATED  # VARCHAR2(200 BYTE)
    VALUENUM: float = AUTOPOPULATED  # NUMBER
    VALUEUOM: str = AUTOPOPULATED  # VARCHAR2(20 BYTE)
    WARNING: int = AUTOPOPULATED  # NUMBER(1,0)
    ERROR: int = AUTOPOPULATED  # NUMBER(1,0)
    RESULTSTATUS: str = AUTOPOPULATED  # VARCHAR2(20 BYTE)
    STOPPED: str = AUTOPOPULATED  # VARCHAR2(20 BYTE)
    
    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, STORETIME, CGID, 
               VALUE, VALUENUM, VALUEUOM, WARNING, ERROR, RESULTSTATUS, STOPPED
        FROM chartevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        print("type of icustay_id: ", type(icustay_id))
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.ITEMID, self.CHARTTIME, \
            self.STORETIME, self.CGID, self.VALUE, self.VALUENUM, self.VALUEUOM, self.WARNING, \
            self.ERROR, self.RESULTSTATUS, self.STOPPED = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "ITEMID": self.ITEMID,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "CGID": self.CGID,
                "VALUE": self.VALUE,
                "VALUENUM": self.VALUENUM,
                "VALUEUOM": self.VALUEUOM,
                "WARNING": self.WARNING,
                "ERROR": self.ERROR,
                "RESULTSTATUS": self.RESULTSTATUS,
                "STOPPED": self.STOPPED
            }
        else:
            result = {}

        cursor.close()
        return result

class CPTEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    COSTCENTER: str = AUTOPOPULATED  # VARCHAR(10)
    CHARTDATE: str = AUTOPOPULATED  # TIMESTAMP(0)
    CPT_CD: str = AUTOPOPULATED  # VARCHAR(10)
    CPT_NUMBER: int = AUTOPOPULATED
    CPT_SUFFIX: str = AUTOPOPULATED  # VARCHAR(5)
    TICKET_ID_SEQ: int = AUTOPOPULATED
    SECTIONHEADER: str = AUTOPOPULATED  # VARCHAR(50)
    SUBSECTIONHEADER: str = AUTOPOPULATED  # VARCHAR(300)
    DESCRIPTION: str = AUTOPOPULATED  # VARCHAR(200)

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, COSTCENTER, CHARTDATE, CPT_CD, CPT_NUMBER, 
               CPT_SUFFIX, TICKET_ID_SEQ, SECTIONHEADER, SUBSECTIONHEADER, DESCRIPTION
        FROM cptevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.COSTCENTER, self.CHARTDATE, \
            self.CPT_CD, self.CPT_NUMBER, self.CPT_SUFFIX, self.TICKET_ID_SEQ, \
            self.SECTIONHEADER, self.SUBSECTIONHEADER, self.DESCRIPTION = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTDATE = datetime.strptime(self.CHARTDATE, '%Y-%m-%d %H:%M:%S') if self.CHARTDATE else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "COSTCENTER": self.COSTCENTER,
                "CHARTDATE": self.CHARTDATE.isoformat() if self.CHARTDATE else None,
                "CPT_CD": self.CPT_CD,
                "CPT_NUMBER": self.CPT_NUMBER,
                "CPT_SUFFIX": self.CPT_SUFFIX,
                "TICKET_ID_SEQ": self.TICKET_ID_SEQ,
                "SECTIONHEADER": self.SECTIONHEADER,
                "SUBSECTIONHEADER": self.SUBSECTIONHEADER,
                "DESCRIPTION": self.DESCRIPTION
            }
        else:
            result = {}

        cursor.close()
        return result

class D_CPT(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    CATEGORY: int = AUTOPOPULATED  # SMALLINT
    SECTIONRANGE: str = AUTOPOPULATED  # VARCHAR(100)
    SECTIONHEADER: str = AUTOPOPULATED  # VARCHAR(50)
    SUBSECTIONRANGE: str = AUTOPOPULATED  # VARCHAR(100)
    SUBSECTIONHEADER: str = AUTOPOPULATED  # VARCHAR(300)
    CODESUFFIX: str = AUTOPOPULATED  # VARCHAR(5)
    MINCODEINSUBSECTION: int = AUTOPOPULATED
    MAXCODEINSUBSECTION: int = AUTOPOPULATED
    def get(self, cpt_cd: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, CATEGORY, SECTIONRANGE, SECTIONHEADER, SUBSECTIONRANGE, SUBSECTIONHEADER, 
               CODESUFFIX, MINCODEINSUBSECTION, MAXCODEINSUBSECTION
        FROM d_cpt
        WHERE ? BETWEEN MINCODEINSUBSECTION AND MAXCODEINSUBSECTION
        """
        cursor.execute(query, (cpt_cd,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.CATEGORY, self.SECTIONRANGE, self.SECTIONHEADER, self.SUBSECTIONRANGE, \
            self.SUBSECTIONHEADER, self.CODESUFFIX, self.MINCODEINSUBSECTION, self.MAXCODEINSUBSECTION = row

            result = {
                "ROW_ID": self.ROW_ID,
                "CATEGORY": self.CATEGORY,
                "SECTIONRANGE": self.SECTIONRANGE,
                "SECTIONHEADER": self.SECTIONHEADER,
                "SUBSECTIONRANGE": self.SUBSECTIONRANGE,
                "SUBSECTIONHEADER": self.SUBSECTIONHEADER,
                "CODESUFFIX": self.CODESUFFIX,
                "MINCODEINSUBSECTION": self.MINCODEINSUBSECTION,
                "MAXCODEINSUBSECTION": self.MAXCODEINSUBSECTION
            }
        else:
            result = {}

        cursor.close()
        return result

class D_ICD_DIAGNOSES(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    ICD9_CODE: str = AUTOPOPULATED  # VARCHAR(10)
    SHORT_TITLE: str = AUTOPOPULATED  # VARCHAR(50)
    LONG_TITLE: str = AUTOPOPULATED  # VARCHAR(300)

    def get(self, icd9_code: str, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
        FROM d_icd_diagnoses
        WHERE ICD9_CODE = ?
        """
        cursor.execute(query, (icd9_code,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.ICD9_CODE, self.SHORT_TITLE, self.LONG_TITLE = row

            result = {
                "ROW_ID": self.ROW_ID,
                "ICD9_CODE": self.ICD9_CODE,
                "SHORT_TITLE": self.SHORT_TITLE,
                "LONG_TITLE": self.LONG_TITLE
            }
        else:
            result = {}

        cursor.close()
        return result

class D_ICD_PROCEDURES(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    ICD9_CODE: str = AUTOPOPULATED  # VARCHAR(10)
    SHORT_TITLE: str = AUTOPOPULATED  # VARCHAR(50)
    LONG_TITLE: str = AUTOPOPULATED  # VARCHAR(300)

    def get(self, icd9_code: str, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
        FROM d_icd_diagnoses
        WHERE ICD9_CODE = ?
        """
        cursor.execute(query, (icd9_code,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.ICD9_CODE, self.SHORT_TITLE, self.LONG_TITLE = row

            result = {
                "ROW_ID": self.ROW_ID,
                "ICD9_CODE": self.ICD9_CODE,
                "SHORT_TITLE": self.SHORT_TITLE,
                "LONG_TITLE": self.LONG_TITLE
            }
        else:
            result = {}

        cursor.close()
        return result

class D_ITEMS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    ITEMID: int = AUTOPOPULATED
    LABEL: str = AUTOPOPULATED  # VARCHAR(200)
    ABBREVIATION: str = AUTOPOPULATED  # VARCHAR(100)
    DBSOURCE: str = AUTOPOPULATED  # VARCHAR(20)
    LINKSTO: str = AUTOPOPULATED  # VARCHAR(50)
    CATEGORY: str = AUTOPOPULATED  # VARCHAR(100)
    UNITNAME: str = AUTOPOPULATED  # VARCHAR(100)
    PARAM_TYPE: str = AUTOPOPULATED  # VARCHAR(30)
    CONCEPTID: int = AUTOPOPULATED

    def get(self, item_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, ITEMID, LABEL, ABBREVIATION, DBSOURCE, LINKSTO, CATEGORY, UNITNAME, 
               PARAM_TYPE, CONCEPTID
        FROM d_items
        WHERE ITEMID = ?
        """
        cursor.execute(query, (item_id,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.ITEMID, self.LABEL, self.ABBREVIATION, self.DBSOURCE, self.LINKSTO, \
            self.CATEGORY, self.UNITNAME, self.PARAM_TYPE, self.CONCEPTID = row

            result = {
                "ROW_ID": self.ROW_ID,
                "ITEMID": self.ITEMID,
                "LABEL": self.LABEL,
                "ABBREVIATION": self.ABBREVIATION,
                "DBSOURCE": self.DBSOURCE,
                "LINKSTO": self.LINKSTO,
                "CATEGORY": self.CATEGORY,
                "UNITNAME": self.UNITNAME,
                "PARAM_TYPE": self.PARAM_TYPE,
                "CONCEPTID": self.CONCEPTID
            }
        else:
            result = {}

        cursor.close()
        return result

class D_LABITEMS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    ITEMID: int = AUTOPOPULATED
    LABEL: str = AUTOPOPULATED  # VARCHAR(100)
    FLUID: str = AUTOPOPULATED  # VARCHAR(100)
    CATEGORY: str = AUTOPOPULATED  # VARCHAR(100)
    LOINC_CODE: str = AUTOPOPULATED  # VARCHAR(100)
    
    def get(self, item_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, ITEMID, LABEL, FLUID, CATEGORY, LOINC_CODE
        FROM d_labitems
        WHERE ITEMID = ?
        """
        cursor.execute(query, (item_id,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.ITEMID, self.LABEL, self.FLUID, self.CATEGORY, self.LOINC_CODE = row

            result = {
                "ROW_ID": self.ROW_ID,
                "ITEMID": self.ITEMID,
                "LABEL": self.LABEL,
                "FLUID": self.FLUID,
                "CATEGORY": self.CATEGORY,
                "LOINC_CODE": self.LOINC_CODE
            }
        else:
            result = {}

        cursor.close()
        return result

class DATETIMEEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    ITEMID: int = AUTOPOPULATED
    CHARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    STORETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CGID: int = AUTOPOPULATED
    VALUE: str = AUTOPOPULATED  # TIMESTAMP(0)
    VALUEUOM: str = AUTOPOPULATED  # VARCHAR(50)
    WARNING: int = AUTOPOPULATED  # SMALLINT
    ERROR: int = AUTOPOPULATED  # SMALLINT
    RESULTSTATUS: str = AUTOPOPULATED  # VARCHAR(50)
    STOPPED: str = AUTOPOPULATED  # VARCHAR(50)

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, STORETIME, CGID, 
               VALUE, VALUEUOM, WARNING, ERROR, RESULTSTATUS, STOPPED
        FROM datetimeevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.ITEMID, self.CHARTTIME, \
            self.STORETIME, self.CGID, self.VALUE, self.VALUEUOM, self.WARNING, self.ERROR, \
            self.RESULTSTATUS, self.STOPPED = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None
            self.VALUE = datetime.strptime(self.VALUE, '%Y-%m-%d %H:%M:%S') if self.VALUE else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "ITEMID": self.ITEMID,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "VALUE": self.VALUE.isoformat() if self.VALUE else None,
                "VALUEUOM": self.VALUEUOM,
                "WARNING": self.WARNING,
                "ERROR": self.ERROR,
                "RESULTSTATUS": self.RESULTSTATUS,
                "STOPPED": self.STOPPED
            }
        else:
            result = {}

        cursor.close()
        return result


class DIAGNOSES_ICD(BaseModel):
    ROW_ID: int = AUTOPOPULATED  # not null
    SUBJECT_ID: int = AUTOPOPULATED  # not null
    HADM_ID: int = AUTOPOPULATED  # not null
    SEQ_NUM: int = AUTOPOPULATED
    ICD9_CODE: str = AUTOPOPULATED  # VARCHAR(10)

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
        FROM diagnoses_icd
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.SEQ_NUM, self.ICD9_CODE = row

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "SEQ_NUM": self.SEQ_NUM,
                "ICD9_CODE": self.ICD9_CODE
            }
        else:
            result = {}

        cursor.close()
        return result


class DRGCODES(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    DRG_TYPE: str = AUTOPOPULATED  # VARCHAR(20)
    DRG_CODE: str = AUTOPOPULATED  # VARCHAR(20)
    DESCRIPTION: str = AUTOPOPULATED  # VARCHAR(300)
    DRG_SEVERITY: int = AUTOPOPULATED  # SMALLINT
    DRG_MORTALITY: int = AUTOPOPULATED  # SMALLINT

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, DRG_TYPE, DRG_CODE, DESCRIPTION, 
               DRG_SEVERITY, DRG_MORTALITY
        FROM drgcodes
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.DRG_TYPE, self.DRG_CODE, \
            self.DESCRIPTION, self.DRG_SEVERITY, self.DRG_MORTALITY = row

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "DRG_TYPE": self.DRG_TYPE,
                "DRG_CODE": self.DRG_CODE,
                "DESCRIPTION": self.DESCRIPTION,
                "DRG_SEVERITY": self.DRG_SEVERITY,
                "DRG_MORTALITY": self.DRG_MORTALITY
            }
        else:
            result = {}

        cursor.close()
        return result

class ICUSTAYS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    DBSOURCE: str = AUTOPOPULATED  # VARCHAR(20)
    FIRST_CAREUNIT: str = AUTOPOPULATED  # VARCHAR(20)
    LAST_CAREUNIT: str = AUTOPOPULATED  # VARCHAR(20)
    FIRST_WARDID: int = AUTOPOPULATED  # SMALLINT
    LAST_WARDID: int = AUTOPOPULATED  # SMALLINT
    INTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    OUTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    LOS: float = AUTOPOPULATED  # DOUBLE

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, DBSOURCE, FIRST_CAREUNIT, LAST_CAREUNIT, 
               FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME, LOS
        FROM icustays
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.DBSOURCE, \
            self.FIRST_CAREUNIT, self.LAST_CAREUNIT, self.FIRST_WARDID, self.LAST_WARDID, \
            self.INTIME, self.OUTTIME, self.LOS = row

            # Convert string timestamps to datetime objects if needed
            self.INTIME = datetime.strptime(self.INTIME, '%Y-%m-%d %H:%M:%S') if self.INTIME else None
            self.OUTTIME = datetime.strptime(self.OUTTIME, '%Y-%m-%d %H:%M:%S') if self.OUTTIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "DBSOURCE": self.DBSOURCE,
                "FIRST_CAREUNIT": self.FIRST_CAREUNIT,
                "LAST_CAREUNIT": self.LAST_CAREUNIT,
                "FIRST_WARDID": self.FIRST_WARDID,
                "LAST_WARDID": self.LAST_WARDID,
                "INTIME": self.INTIME.isoformat() if self.INTIME else None,
                "OUTTIME": self.OUTTIME.isoformat() if self.OUTTIME else None,
                "LOS": self.LOS
            }
        else:
            result = {}

        cursor.close()
        return result

class INPUTEVENTS_CV(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    CHARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ITEMID: int = AUTOPOPULATED
    AMOUNT: float = AUTOPOPULATED  # DOUBLE PRECISION
    AMOUNTUOM: str = AUTOPOPULATED  # VARCHAR(30)
    RATE: float = AUTOPOPULATED  # DOUBLE PRECISION
    RATEUOM: str = AUTOPOPULATED  # VARCHAR(30)
    STORETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CGID: int = AUTOPOPULATED  # BIGINT
    ORDERID: int = AUTOPOPULATED  # BIGINT
    LINKORDERID: int = AUTOPOPULATED  # BIGINT
    STOPPED: str = AUTOPOPULATED  # VARCHAR(30)
    NEWBOTTLE: int = AUTOPOPULATED
    ORIGINALAMOUNT: float = AUTOPOPULATED  # DOUBLE PRECISION
    ORIGINALAMOUNTUOM: str = AUTOPOPULATED  # VARCHAR(30)
    ORIGINALROUTE: str = AUTOPOPULATED  # VARCHAR(30)
    ORIGINALRATE: float = AUTOPOPULATED  # DOUBLE PRECISION
    ORIGINALRATEUOM: str = AUTOPOPULATED  # VARCHAR(30)
    ORIGINALSITE: str = AUTOPOPULATED  # VARCHAR(30)

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT, AMOUNTUOM, 
               RATE, RATEUOM, STORETIME, CGID, ORDERID, LINKORDERID, STOPPED, NEWBOTTLE, 
               ORIGINALAMOUNT, ORIGINALAMOUNTUOM, ORIGINALROUTE, ORIGINALRATE, 
               ORIGINALRATEUOM, ORIGINALSITE
        FROM inputevents_cv
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.CHARTTIME, \
            self.ITEMID, self.AMOUNT, self.AMOUNTUOM, self.RATE, self.RATEUOM, self.STORETIME, \
            self.CGID, self.ORDERID, self.LINKORDERID, self.STOPPED, self.NEWBOTTLE, \
            self.ORIGINALAMOUNT, self.ORIGINALAMOUNTUOM, self.ORIGINALROUTE, self.ORIGINALRATE, \
            self.ORIGINALRATEUOM, self.ORIGINALSITE = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "ITEMID": self.ITEMID,
                "AMOUNT": self.AMOUNT,
                "AMOUNTUOM": self.AMOUNTUOM,
                "RATE": self.RATE,
                "RATEUOM": self.RATEUOM,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "CGID": self.CGID,
                "ORDERID": self.ORDERID,
                "LINKORDERID": self.LINKORDERID,
                "STOPPED": self.STOPPED,
                "NEWBOTTLE": self.NEWBOTTLE,
                "ORIGINALAMOUNT": self.ORIGINALAMOUNT,
                "ORIGINALAMOUNTUOM": self.ORIGINALAMOUNTUOM,
                "ORIGINALROUTE": self.ORIGINALROUTE,
                "ORIGINALRATE": self.ORIGINALRATE,
                "ORIGINALRATEUOM": self.ORIGINALRATEUOM,
                "ORIGINALSITE": self.ORIGINALSITE
            }
        else:
            result = {}

        cursor.close()
        return result


class INPUTEVENTS_MV(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    STARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ENDTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ITEMID: int = AUTOPOPULATED
    AMOUNT: float = AUTOPOPULATED  # DOUBLE PRECISION
    AMOUNTUOM: str = AUTOPOPULATED  # VARCHAR(30)
    RATE: float = AUTOPOPULATED  # DOUBLE PRECISION
    RATEUOM: str = AUTOPOPULATED  # VARCHAR(30)
    STORETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CGID: int = AUTOPOPULATED  # BIGINT
    ORDERID: int = AUTOPOPULATED  # BIGINT
    LINKORDERID: int = AUTOPOPULATED  # BIGINT
    ORDERCATEGORYNAME: str = AUTOPOPULATED  # VARCHAR(100)
    SECONDARYORDERCATEGORYNAME: str = AUTOPOPULATED  # VARCHAR(100)
    ORDERCOMPONENTTYPEDESCRIPTION: str = AUTOPOPULATED  # VARCHAR(200)
    ORDERCATEGORYDESCRIPTION: str = AUTOPOPULATED  # VARCHAR(50)
    PATIENTWEIGHT: float = AUTOPOPULATED  # DOUBLE PRECISION
    TOTALAMOUNT: float = AUTOPOPULATED  # DOUBLE PRECISION
    TOTALAMOUNTUOM: str = AUTOPOPULATED  # VARCHAR(50)
    ISOPENBAG: int = AUTOPOPULATED  # SMALLINT
    CONTINUEINNEXTDEPT: int = AUTOPOPULATED  # SMALLINT
    CANCELREASON: int = AUTOPOPULATED  # SMALLINT
    STATUSDESCRIPTION: str = AUTOPOPULATED  # VARCHAR(30)
    COMMENTS_STATUS: str = AUTOPOPULATED  # VARCHAR(30)
    COMMENTS_TITLE: str = AUTOPOPULATED  # VARCHAR(100)
    COMMENTS_DATE: str = AUTOPOPULATED  # TIMESTAMP(0)
    ORIGINALAMOUNT: float = AUTOPOPULATED  # DOUBLE PRECISION
    ORIGINALRATE: float = AUTOPOPULATED  # DOUBLE PRECISION

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, STARTTIME, ENDTIME, ITEMID, AMOUNT, AMOUNTUOM, 
               RATE, RATEUOM, STORETIME, CGID, ORDERID, LINKORDERID, ORDERCATEGORYNAME, 
               SECONDARYORDERCATEGORYNAME, ORDERCOMPONENTTYPEDESCRIPTION, ORDERCATEGORYDESCRIPTION, 
               PATIENTWEIGHT, TOTALAMOUNT, TOTALAMOUNTUOM, ISOPENBAG, CONTINUEINNEXTDEPT, 
               CANCELREASON, STATUSDESCRIPTION, COMMENTS_STATUS, COMMENTS_TITLE, COMMENTS_DATE, 
               ORIGINALAMOUNT, ORIGINALRATE
        FROM inputevents_mv
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.STARTTIME, self.ENDTIME, \
            self.ITEMID, self.AMOUNT, self.AMOUNTUOM, self.RATE, self.RATEUOM, self.STORETIME, \
            self.CGID, self.ORDERID, self.LINKORDERID, self.ORDERCATEGORYNAME, self.SECONDARYORDERCATEGORYNAME, \
            self.ORDERCOMPONENTTYPEDESCRIPTION, self.ORDERCATEGORYDESCRIPTION, self.PATIENTWEIGHT, \
            self.TOTALAMOUNT, self.TOTALAMOUNTUOM, self.ISOPENBAG, self.CONTINUEINNEXTDEPT, self.CANCELREASON, \
            self.STATUSDESCRIPTION, self.COMMENTS_STATUS, self.COMMENTS_TITLE, self.COMMENTS_DATE, \
            self.ORIGINALAMOUNT, self.ORIGINALRATE = row

            # Convert string timestamps to datetime objects if needed
            self.STARTTIME = datetime.strptime(self.STARTTIME, '%Y-%m-%d %H:%M:%S') if self.STARTTIME else None
            self.ENDTIME = datetime.strptime(self.ENDTIME, '%Y-%m-%d %H:%M:%S') if self.ENDTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None
            self.COMMENTS_DATE = datetime.strptime(self.COMMENTS_DATE, '%Y-%m-%d %H:%M:%S') if self.COMMENTS_DATE else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "STARTTIME": self.STARTTIME.isoformat() if self.STARTTIME else None,
                "ENDTIME": self.ENDTIME.isoformat() if self.ENDTIME else None,
                "ITEMID": self.ITEMID,
                "AMOUNT": self.AMOUNT,
                "AMOUNTUOM": self.AMOUNTUOM,
                "RATE": self.RATE,
                "RATEUOM": self.RATEUOM,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "CGID": self.CGID,
                "ORDERID": self.ORDERID,
                "LINKORDERID": self.LINKORDERID,
                "ORDERCATEGORYNAME": self.ORDERCATEGORYNAME,
                "SECONDARYORDERCATEGORYNAME": self.SECONDARYORDERCATEGORYNAME,
                "ORDERCOMPONENTTYPEDESCRIPTION": self.ORDERCOMPONENTTYPEDESCRIPTION,
                "ORDERCATEGORYDESCRIPTION": self.ORDERCATEGORYDESCRIPTION,
                "PATIENTWEIGHT": self.PATIENTWEIGHT,
                "TOTALAMOUNT": self.TOTALAMOUNT,
                "TOTALAMOUNTUOM": self.TOTALAMOUNTUOM,
                "ISOPENBAG": self.ISOPENBAG,
                "CONTINUEINNEXTDEPT": self.CONTINUEINNEXTDEPT,
                "CANCELREASON": self.CANCELREASON,
                "STATUSDESCRIPTION": self.STATUSDESCRIPTION,
                "COMMENTS_STATUS": self.COMMENTS_STATUS,
                "COMMENTS_TITLE": self.COMMENTS_TITLE,
                "COMMENTS_DATE": self.COMMENTS_DATE.isoformat() if self.COMMENTS_DATE else None,
                "ORIGINALAMOUNT": self.ORIGINALAMOUNT,
                "ORIGINALRATE": self.ORIGINALRATE
            }
        else:
            result = {}

        cursor.close()
        return result

class LABEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ITEMID: int = AUTOPOPULATED
    CHARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    VALUE: str = AUTOPOPULATED  # VARCHAR(200)
    VALUENUM: float = AUTOPOPULATED  # DOUBLE PRECISION
    VALUEUOM: str = AUTOPOPULATED  # VARCHAR(20)
    FLAG: str = AUTOPOPULATED  # VARCHAR(20)

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM, FLAG
        FROM labevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ITEMID, self.CHARTTIME, \
            self.VALUE, self.VALUENUM, self.VALUEUOM, self.FLAG = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ITEMID": self.ITEMID,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "VALUE": self.VALUE,
                "VALUENUM": self.VALUENUM,
                "VALUEUOM": self.VALUEUOM,
                "FLAG": self.FLAG
            }
        else:
            result = {}

        cursor.close()
        return result

class MICROBIOLOGYEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    CHARTDATE: str = AUTOPOPULATED  # TIMESTAMP(0)
    CHARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    SPEC_ITEMID: int = AUTOPOPULATED
    SPEC_TYPE_DESC: str = AUTOPOPULATED  # VARCHAR(100)
    ORG_ITEMID: int = AUTOPOPULATED
    ORG_NAME: str = AUTOPOPULATED  # VARCHAR(100)
    ISOLATE_NUM: int = AUTOPOPULATED  # SMALLINT
    AB_ITEMID: int = AUTOPOPULATED
    AB_NAME: str = AUTOPOPULATED  # VARCHAR(30)
    DILUTION_TEXT: str = AUTOPOPULATED  # VARCHAR(10)
    DILUTION_COMPARISON: str = AUTOPOPULATED  # VARCHAR(20)
    DILUTION_VALUE: float = AUTOPOPULATED  # DOUBLE PRECISION
    INTERPRETATION: str = AUTOPOPULATED  # VARCHAR(5)



    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, SPEC_ITEMID, SPEC_TYPE_DESC, 
               ORG_ITEMID, ORG_NAME, ISOLATE_NUM, AB_ITEMID, AB_NAME, DILUTION_TEXT, 
               DILUTION_COMPARISON, DILUTION_VALUE, INTERPRETATION
        FROM microbiologyevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.CHARTDATE, self.CHARTTIME, self.SPEC_ITEMID, \
            self.SPEC_TYPE_DESC, self.ORG_ITEMID, self.ORG_NAME, self.ISOLATE_NUM, self.AB_ITEMID, \
            self.AB_NAME, self.DILUTION_TEXT, self.DILUTION_COMPARISON, self.DILUTION_VALUE, \
            self.INTERPRETATION = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTDATE = datetime.strptime(self.CHARTDATE, '%Y-%m-%d %H:%M:%S') if self.CHARTDATE else None
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "CHARTDATE": self.CHARTDATE.isoformat() if self.CHARTDATE else None,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "SPEC_ITEMID": self.SPEC_ITEMID,
                "SPEC_TYPE_DESC": self.SPEC_TYPE_DESC,
                "ORG_ITEMID": self.ORG_ITEMID,
                "ORG_NAME": self.ORG_NAME,
                "ISOLATE_NUM": self.ISOLATE_NUM,
                "AB_ITEMID": self.AB_ITEMID,
                "AB_NAME": self.AB_NAME,
                "DILUTION_TEXT": self.DILUTION_TEXT,
                "DILUTION_COMPARISON": self.DILUTION_COMPARISON,
                "DILUTION_VALUE": self.DILUTION_VALUE,
                "INTERPRETATION": self.INTERPRETATION
            }
        else:
            result = {}

        cursor.close()
        return result

class NOTEEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    CHARTDATE: str = AUTOPOPULATED  # TIMESTAMP(0)
    CHARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    STORETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CATEGORY: str = AUTOPOPULATED  # VARCHAR(50)
    DESCRIPTION: str = AUTOPOPULATED  # VARCHAR(300)
    CGID: int = AUTOPOPULATED
    ISERROR: str = AUTOPOPULATED  # CHAR(1)
    TEXT: str = AUTOPOPULATED  # TEXT


    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME, CATEGORY, 
               DESCRIPTION, CGID, ISERROR, TEXT
        FROM noteevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.CHARTDATE, self.CHARTTIME, \
            self.STORETIME, self.CATEGORY, self.DESCRIPTION, self.CGID, self.ISERROR, \
            self.TEXT = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTDATE = datetime.strptime(self.CHARTDATE, '%Y-%m-%d %H:%M:%S') if self.CHARTDATE else None
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "CHARTDATE": self.CHARTDATE.isoformat() if self.CHARTDATE else None,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "CATEGORY": self.CATEGORY,
                "DESCRIPTION": self.DESCRIPTION,
                "CGID": self.CGID,
                "ISERROR": self.ISERROR,
                "TEXT": self.TEXT
            }
        else:
            result = {}

        cursor.close()
        return result

class OUTPUTEVENTS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    CHARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ITEMID: int = AUTOPOPULATED
    VALUE: float = AUTOPOPULATED  # DOUBLE PRECISION
    VALUEUOM: str = AUTOPOPULATED  # VARCHAR(30)
    STORETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CGID: int = AUTOPOPULATED  # BIGINT
    STOPPED: str = AUTOPOPULATED  # VARCHAR(30)
    NEWBOTTLE: int = AUTOPOPULATED
    ISERROR: int = AUTOPOPULATED  # SMALLINT

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE, VALUEUOM, 
               STORETIME, CGID, STOPPED, NEWBOTTLE, ISERROR
        FROM outputevents
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.CHARTTIME, \
            self.ITEMID, self.VALUE, self.VALUEUOM, self.STORETIME, self.CGID, self.STOPPED, \
            self.NEWBOTTLE, self.ISERROR = row

            # Convert string timestamps to datetime objects if needed
            self.CHARTTIME = datetime.strptime(self.CHARTTIME, '%Y-%m-%d %H:%M:%S') if self.CHARTTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "CHARTTIME": self.CHARTTIME.isoformat() if self.CHARTTIME else None,
                "ITEMID": self.ITEMID,
                "VALUE": self.VALUE,
                "VALUEUOM": self.VALUEUOM,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "CGID": self.CGID,
                "STOPPED": self.STOPPED,
                "NEWBOTTLE": self.NEWBOTTLE,
                "ISERROR": self.ISERROR
            }
        else:
            result = {}

        cursor.close()
        return result


class PATIENTS(BaseModel):
    # patient_name: str = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED 
    GENDER: int = AUTOPOPULATED 
    DOB: str = AUTOPOPULATED
    DOD: str = AUTOPOPULATED
    DOD_HOSP: str = AUTOPOPULATED
    DOD_SSN: str = AUTOPOPULATED
    EXPIRE_FLAG: int = AUTOPOPULATED


    def get(self, subject_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT SUBJECT_ID, GENDER, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG
        FROM patients
        WHERE SUBJECT_ID = ?
        """
        cursor.execute(query, (subject_id,))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.SUBJECT_ID, self.GENDER, self.DOB, self.DOD, self.DOD_HOSP, self.DOD_SSN, self.EXPIRE_FLAG = row

            # Convert string dates to datetime objects if needed
            self.DOB = datetime.strptime(self.DOB, '%Y-%m-%d %H:%M:%S') if self.DOB else None
            self.DOD = datetime.strptime(self.DOD, '%Y-%m-%d %H:%M:%S') if self.DOD else None
            self.DOD_HOSP = datetime.strptime(self.DOD_HOSP, '%Y-%m-%d %H:%M:%S') if self.DOD_HOSP else None
            self.DOD_SSN = datetime.strptime(self.DOD_SSN, '%Y-%m-%d %H:%M:%S') if self.DOD_SSN else None

            result = {
                "SUBJECT_ID": self.SUBJECT_ID,
                "GENDER": self.GENDER,
                "DOB": self.DOB.isoformat() if self.DOB else None,
                "DOD": self.DOD.isoformat() if self.DOD else None,
                "DOD_HOSP": self.DOD_HOSP.isoformat() if self.DOD_HOSP else None,
                "DOD_SSN": self.DOD_SSN.isoformat() if self.DOD_SSN else None,
                "EXPIRE_FLAG": self.EXPIRE_FLAG
            }
        else:
            result = {}

        cursor.close()
        return result

class PRESCRIPTIONS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    STARTDATE: str = AUTOPOPULATED  # TIMESTAMP(0)
    ENDDATE: str = AUTOPOPULATED  # TIMESTAMP(0)
    DRUG_TYPE: str = AUTOPOPULATED  # VARCHAR(100)
    DRUG: str = AUTOPOPULATED  # VARCHAR(100)
    DRUG_NAME_POE: str = AUTOPOPULATED  # VARCHAR(100)
    DRUG_NAME_GENERIC: str = AUTOPOPULATED  # VARCHAR(100)
    FORMULARY_DRUG_CD: str = AUTOPOPULATED  # VARCHAR(120)
    GSN: str = AUTOPOPULATED  # VARCHAR(200)
    NDC: str = AUTOPOPULATED  # VARCHAR(120)
    PROD_STRENGTH: str = AUTOPOPULATED  # VARCHAR(120)
    DOSE_VAL_RX: str = AUTOPOPULATED  # VARCHAR(120)
    DOSE_UNIT_RX: str = AUTOPOPULATED  # VARCHAR(120)
    FORM_VAL_DISP: str = AUTOPOPULATED  # VARCHAR(120)
    FORM_UNIT_DISP: str = AUTOPOPULATED  # VARCHAR(120)
    ROUTE: str = AUTOPOPULATED  # VARCHAR(120)

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, STARTDATE, ENDDATE, DRUG_TYPE, DRUG, 
               DRUG_NAME_POE, DRUG_NAME_GENERIC, FORMULARY_DRUG_CD, GSN, NDC, PROD_STRENGTH, 
               DOSE_VAL_RX, DOSE_UNIT_RX, FORM_VAL_DISP, FORM_UNIT_DISP, ROUTE
        FROM prescriptions
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.STARTDATE, self.ENDDATE, \
            self.DRUG_TYPE, self.DRUG, self.DRUG_NAME_POE, self.DRUG_NAME_GENERIC, self.FORMULARY_DRUG_CD, \
            self.GSN, self.NDC, self.PROD_STRENGTH, self.DOSE_VAL_RX, self.DOSE_UNIT_RX, \
            self.FORM_VAL_DISP, self.FORM_UNIT_DISP, self.ROUTE = row

            # Convert string timestamps to datetime objects if needed
            self.STARTDATE = datetime.strptime(self.STARTDATE, '%Y-%m-%d %H:%M:%S') if self.STARTDATE else None
            self.ENDDATE = datetime.strptime(self.ENDDATE, '%Y-%m-%d %H:%M:%S') if self.ENDDATE else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "STARTDATE": self.STARTDATE.isoformat() if self.STARTDATE else None,
                "ENDDATE": self.ENDDATE.isoformat() if self.ENDDATE else None,
                "DRUG_TYPE": self.DRUG_TYPE,
                "DRUG": self.DRUG,
                "DRUG_NAME_POE": self.DRUG_NAME_POE,
                "DRUG_NAME_GENERIC": self.DRUG_NAME_GENERIC,
                "FORMULARY_DRUG_CD": self.FORMULARY_DRUG_CD,
                "GSN": self.GSN,
                "NDC": self.NDC,
                "PROD_STRENGTH": self.PROD_STRENGTH,
                "DOSE_VAL_RX": self.DOSE_VAL_RX,
                "DOSE_UNIT_RX": self.DOSE_UNIT_RX,
                "FORM_VAL_DISP": self.FORM_VAL_DISP,
                "FORM_UNIT_DISP": self.FORM_UNIT_DISP,
                "ROUTE": self.ROUTE
            }
        else:
            result = {}

        cursor.close()
        return result


class PROCEDUREEVENTS_MV(BaseModel):
    ROW_ID: int = AUTOPOPULATED  # NOT NULL
    SUBJECT_ID: int = AUTOPOPULATED  # NOT NULL
    HADM_ID: int = AUTOPOPULATED  # NOT NULL
    ICUSTAY_ID: int = AUTOPOPULATED
    STARTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ENDTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    ITEMID: int = AUTOPOPULATED
    VALUE: float = AUTOPOPULATED  # DOUBLE PRECISION
    VALUEUOM: str = AUTOPOPULATED  # VARCHAR(30)
    LOCATION: str = AUTOPOPULATED  # VARCHAR(30)
    LOCATIONCATEGORY: str = AUTOPOPULATED  # VARCHAR(30)
    STORETIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    CGID: int = AUTOPOPULATED
    ORDERID: int = AUTOPOPULATED
    LINKORDERID: int = AUTOPOPULATED
    ORDERCATEGORYNAME: str = AUTOPOPULATED  # VARCHAR(100)
    SECONDARYORDERCATEGORYNAME: str = AUTOPOPULATED  # VARCHAR(100)
    ORDERCATEGORYDESCRIPTION: str = AUTOPOPULATED  # VARCHAR(50)
    ISOPENBAG: int = AUTOPOPULATED  # SMALLINT
    CONTINUEINNEXTDEPT: int = AUTOPOPULATED  # SMALLINT
    CANCELREASON: int = AUTOPOPULATED  # SMALLINT
    STATUSDESCRIPTION: str = AUTOPOPULATED  # VARCHAR(30)
    COMMENTS_EDITEDBY: str = AUTOPOPULATED  # VARCHAR(30)
    COMMENTS_CANCELEDBY: str = AUTOPOPULATED  # VARCHAR(30)
    COMMENTS_DATE: str = AUTOPOPULATED  # TIMESTAMP(0)

    def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, STARTTIME, ENDTIME, ITEMID, VALUE, VALUEUOM, 
               LOCATION, LOCATIONCATEGORY, STORETIME, CGID, ORDERID, LINKORDERID, ORDERCATEGORYNAME, 
               SECONDARYORDERCATEGORYNAME, ORDERCATEGORYDESCRIPTION, ISOPENBAG, CONTINUEINNEXTDEPT, 
               CANCELREASON, STATUSDESCRIPTION, COMMENTS_EDITEDBY, COMMENTS_CANCELEDBY, COMMENTS_DATE
        FROM procedureevents_mv
        WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id, icustay_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.STARTTIME, self.ENDTIME, \
            self.ITEMID, self.VALUE, self.VALUEUOM, self.LOCATION, self.LOCATIONCATEGORY, self.STORETIME, \
            self.CGID, self.ORDERID, self.LINKORDERID, self.ORDERCATEGORYNAME, self.SECONDARYORDERCATEGORYNAME, \
            self.ORDERCATEGORYDESCRIPTION, self.ISOPENBAG, self.CONTINUEINNEXTDEPT, self.CANCELREASON, \
            self.STATUSDESCRIPTION, self.COMMENTS_EDITEDBY, self.COMMENTS_CANCELEDBY, self.COMMENTS_DATE = row

            # Convert string timestamps to datetime objects if needed
            self.STARTTIME = datetime.strptime(self.STARTTIME, '%Y-%m-%d %H:%M:%S') if self.STARTTIME else None
            self.ENDTIME = datetime.strptime(self.ENDTIME, '%Y-%m-%d %H:%M:%S') if self.ENDTIME else None
            self.STORETIME = datetime.strptime(self.STORETIME, '%Y-%m-%d %H:%M:%S') if self.STORETIME else None
            self.COMMENTS_DATE = datetime.strptime(self.COMMENTS_DATE, '%Y-%m-%d %H:%M:%S') if self.COMMENTS_DATE else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "ICUSTAY_ID": self.ICUSTAY_ID,
                "STARTTIME": self.STARTTIME.isoformat() if self.STARTTIME else None,
                "ENDTIME": self.ENDTIME.isoformat() if self.ENDTIME else None,
                "ITEMID": self.ITEMID,
                "VALUE": self.VALUE,
                "VALUEUOM": self.VALUEUOM,
                "LOCATION": self.LOCATION,
                "LOCATIONCATEGORY": self.LOCATIONCATEGORY,
                "STORETIME": self.STORETIME.isoformat() if self.STORETIME else None,
                "CGID": self.CGID,
                "ORDERID": self.ORDERID,
                "LINKORDERID": self.LINKORDERID,
                "ORDERCATEGORYNAME": self.ORDERCATEGORYNAME,
                "SECONDARYORDERCATEGORYNAME": self.SECONDARYORDERCATEGORYNAME,
                "ORDERCATEGORYDESCRIPTION": self.ORDERCATEGORYDESCRIPTION,
                "ISOPENBAG": self.ISOPENBAG,
                "CONTINUEINNEXTDEPT": self.CONTINUEINNEXTDEPT,
                "CANCELREASON": self.CANCELREASON,
                "STATUSDESCRIPTION": self.STATUSDESCRIPTION,
                "COMMENTS_EDITEDBY": self.COMMENTS_EDITEDBY,
                "COMMENTS_CANCELEDBY": self.COMMENTS_CANCELEDBY,
                "COMMENTS_DATE": self.COMMENTS_DATE.isoformat() if self.COMMENTS_DATE else None
            }
        else:
            result = {}

        cursor.close()
        return result    

class PROCEDURES_ICD(BaseModel):
    ROW_ID: int = AUTOPOPULATED  # not null
    SUBJECT_ID: int = AUTOPOPULATED  # not null
    HADM_ID: int = AUTOPOPULATED  # not null
    SEQ_NUM: int = AUTOPOPULATED
    ICD9_CODE: str = AUTOPOPULATED  # VARCHAR(10)

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
        FROM procedures_icd
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.SEQ_NUM, self.ICD9_CODE = row

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "SEQ_NUM": self.SEQ_NUM,
                "ICD9_CODE": self.ICD9_CODE
            }
        else:
            result = {}

        cursor.close()
        return result

class SERVICES(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    TRANSFERTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    PREV_SERVICE: str = AUTOPOPULATED  # VARCHAR(20)
    CURR_SERVICE: str = AUTOPOPULATED  # VARCHAR(20)

    def get(self, subject_id: int, admission_id: int, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, TRANSFERTIME, PREV_SERVICE, CURR_SERVICE
        FROM services
        WHERE SUBJECT_ID = ? AND HADM_ID = ?
        """
        cursor.execute(query, (subject_id, admission_id))
        row = cursor.fetchone()
        # print('row:', row)
        if row:
            self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.TRANSFERTIME, self.PREV_SERVICE, \
            self.CURR_SERVICE = row

            # Convert string timestamps to datetime objects if needed
            self.TRANSFERTIME = datetime.strptime(self.TRANSFERTIME, '%Y-%m-%d %H:%M:%S') if self.TRANSFERTIME else None

            result = {
                "ROW_ID": self.ROW_ID,
                "SUBJECT_ID": self.SUBJECT_ID,
                "HADM_ID": self.HADM_ID,
                "TRANSFERTIME": self.TRANSFERTIME.isoformat() if self.TRANSFERTIME else None,
                "PREV_SERVICE": self.PREV_SERVICE,
                "CURR_SERVICE": self.CURR_SERVICE
            }
        else:
            result = {}

        cursor.close()
        return result

class TRANSFERS(BaseModel):
    ROW_ID: int = AUTOPOPULATED
    SUBJECT_ID: int = AUTOPOPULATED
    HADM_ID: int = AUTOPOPULATED
    ICUSTAY_ID: int = AUTOPOPULATED
    DBSOURCE: str = AUTOPOPULATED  # VARCHAR(20)
    EVENTTYPE: str = AUTOPOPULATED  # VARCHAR(20)
    PREV_CAREUNIT: str = AUTOPOPULATED  # VARCHAR(20)
    CURR_CAREUNIT: str = AUTOPOPULATED  # VARCHAR(20)
    PREV_WARDID: int = AUTOPOPULATED  # SMALLINT
    CURR_WARDID: int = AUTOPOPULATED  # SMALLINT
    INTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    OUTTIME: str = AUTOPOPULATED  # TIMESTAMP(0)
    LOS: int = AUTOPOPULATED  # INT


def get(self, subject_id: int, admission_id: int, icustay_id: int, conn: sqlite3.Connection) -> str:
    cursor = conn.cursor()
    query = """
    SELECT ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, DBSOURCE, EVENTTYPE, PREV_CAREUNIT, 
           CURR_CAREUNIT, PREV_WARDID, CURR_WARDID, INTIME, OUTTIME, LOS
    FROM transfers
    WHERE SUBJECT_ID = ? AND HADM_ID = ? AND ICUSTAY_ID = ?
    """
    cursor.execute(query, (subject_id, admission_id, icustay_id))
    row = cursor.fetchone()
    # print('row:', row)
    if row:
        self.ROW_ID, self.SUBJECT_ID, self.HADM_ID, self.ICUSTAY_ID, self.DBSOURCE, self.EVENTTYPE, \
        self.PREV_CAREUNIT, self.CURR_CAREUNIT, self.PREV_WARDID, self.CURR_WARDID, self.INTIME, \
        self.OUTTIME, self.LOS = row

        # Convert string timestamps to datetime objects if needed
        self.INTIME = datetime.strptime(self.INTIME, '%Y-%m-%d %H:%M:%S') if self.INTIME else None
        self.OUTTIME = datetime.strptime(self.OUTTIME, '%Y-%m-%d %H:%M:%S') if self.OUTTIME else None

        result = {
            "ROW_ID": self.ROW_ID,
            "SUBJECT_ID": self.SUBJECT_ID,
            "HADM_ID": self.HADM_ID,
            "ICUSTAY_ID": self.ICUSTAY_ID,
            "DBSOURCE": self.DBSOURCE,
            "EVENTTYPE": self.EVENTTYPE,
            "PREV_CAREUNIT": self.PREV_CAREUNIT,
            "CURR_CAREUNIT": self.CURR_CAREUNIT,
            "PREV_WARDID": self.PREV_WARDID,
            "CURR_WARDID": self.CURR_WARDID,
            "INTIME": self.INTIME.isoformat() if self.INTIME else None,
            "OUTTIME": self.OUTTIME.isoformat() if self.OUTTIME else None,
            "LOS": self.LOS
        }
    else:
        result = {}

    cursor.close()
    return result

if __name__ == '__main__':
    print('start test')
    PATH = "../physionet.org/files/mimiciii/1.4/mimic3.db"
    
    conn = sqlite3.connect(PATH)
    cursor = sqlite3.connect(PATH).cursor()
    print('Connected Successfully')

    subject_id, admission_id = 22, 165315
    icu_query = """
        SELECT ICUSTAY_ID
        FROM icustays
        WHERE SUBJECT_ID = ?
        AND HADM_ID = ?

    """    
    #admission_query = """
    #    SELECT ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME,
    #    ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE,
    #    LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, EDREGTIME, EDOUTTIME,
    #    DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_CHARTEVENTS_DATA
    #    FROM admissions
    #    LIMIT 1
    #    """
    #query_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    #cursor.execute(admission_query)
    #row = cursor.fetchall()
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
            print("chartevents[icustay_id]: ", type(chartevents[icustay_id]))
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



    print("admission: ", admission)
    print("callout: ", callout)
    print("icustayids: ", icustay_ids)
    print("chartevents: ", chartevents)
    print("cptevents: ", cptevents)
    print("datetimeevents: ", datetimeevents)
    print("diagnoses_icd: ", diagnoses_icd)
    print("drgcodes: ", drgcodes)
    print("icustays: ", icustays)
    print("inputevents_cv: ", inputevents_cv)
    print("inputevents_mv: ", inputevents_mv)
    print("labevents: ", labevents)
    print("microbiologyevents: ", microbiologyevents)
    print("noteevents: ", noteevents)
    print("outputevents: ", outputevents)
    print("patients: ", patients)
    print("prescriptions: ", prescriptions)
    print("procedures_icd: ", procedures_icd)
    print("procedureevents_mv: ", procedureevents_mv)
    print("services: ", services)
    print("transfers: ", transfers)

#print("ADMISSION:",admission)
    #print("fetch result:", admission.get(22, 165315, conn))
