"""Compact item mappings used by the legacy forecast extraction.

These are the item/name selections from the former step_1 implementation. Keeping
mappings separate from event rows lets extraction discover the schema without loading
the event tables into memory.
"""

from collections import defaultdict

import pandas as pd


CHART_ITEMS = {
    "DBP": [8368, 220051, 225310, 8555, 8441, 220180, 8502, 8440, 8503, 8504, 8507, 8506, 224643, 227242],
    "SBP": [51, 220050, 225309, 6701, 455, 220179, 3313, 3315, 442, 3317, 3323, 3321, 224167, 227243],
    "MBP": [52, 220052, 225312, 224, 6702, 224322, 456, 220181, 3312, 3314, 3316, 3322, 3320, 443],
    "GCS_eye": [184, 220739],
    "GCS_motor": [454, 223901],
    "GCS_verbal": [723, 223900],
    "HR": [211, 220045],
    "RR": [618, 220210, 3603, 224689, 614, 651, 224422, 615, 224690, 619, 224688, 227860, 227918],
    "Temperature": [3655, 677, 676, 223762, 223761, 678, 679, 3654],
    "Weight": [224639, 226512, 226846, 763, 226531],
    "Height": [1394, 226707, 226730],
    "FiO2": [3420, 223835, 3422, 189, 727, 190],
    "CRR": [3348, 115, 8377, 224308, 223951],
    "Glucose (Blood)": [225664, 1529, 811, 807, 3745, 50809],
    "Glucose (Whole Blood)": [226537],
    "Glucose (Serum)": [220621, 50931],
    "Bilirubin (Total)": [50885],
    "Bilirubin (Direct)": [50883],
    "Bilirubin (Indirect)": [50884],
    "SpO2_peripheral": [220277],
    "O2_saturation_chart_unclassified": [834, 8498, 220227, 646],
}

LAB_ITEMS = {
    "SO2_bloodgas": [50817],
    "Glucose (Blood)": [225664, 1529, 811, 807, 3745, 50809],
    "Glucose (Whole Blood)": [226537],
    "Glucose (Serum)": [220621, 50931],
    "Bilirubin (Total)": [50885],
    "Bilirubin (Direct)": [50883],
    "Bilirubin (Indirect)": [50884],
    "Sodium": [50983, 50824],
    "Potassium": [50971, 50822],
    "Magnesium": [50960],
    "Phosphate": [50970],
    "Calcium Total": [50893],
    "Calcium Free": [50808],
    "WBC": [51301, 51300],
    "Hct": [50810, 51221],
    "Hgb": [51222, 50811],
    "Chloride": [50902, 50806],
    "Bicarbonate": [50882, 50803],
    "ALT": [50861],
    "ALP": [50863],
    "AST": [50878],
    "Albumin": [50862],
    "Lactate": [50813],
    "LDH": [50954],
    "SG Urine": [51498],
    "pH Urine": [51491, 51094, 220734, 1495, 1880, 1352, 6754, 7262],
    "pH Blood": [50820],
    "PO2": [50821],
    "PCO2": [50818],
    "Total CO2": [50804],
    "Base Excess": [50802],
    "Monocytes": [51254],
    "Basophils": [51146],
    "Eoisinophils": [51200],
    "Neutrophils": [51256],
    "Lymphocytes": [51244, 51245],
    "Lymphocytes (Absolute)": [51133],
    "PT": [51274],
    "PTT": [51275],
    "INR": [51237],
    "Anion Gap": [50868],
    "BUN": [51006],
    "Creatinine Blood": [50912],
    "Creatinine Urine": [51082],
    "MCH": [51248],
    "MCHC": [51249],
    "MCV": [51250],
    "RDW": [51277],
    "Platelet Count": [51265],
    "RBC": [51279],
    "Intubated": [50812],
}

OUTPUT_FIXED_ITEMS = {
    "Ultrafiltrate": [40286],
    "Chest Tube": [226593, 226590, 226591, 226595, 226592],
    "Gastric": [40059, 40052, 226576, 226575, 226573, 40051, 226630],
    "EBL": [40064, 226626, 40491, 226629],
    "Emesis": [40067, 226571, 40490, 41015, 40427],
    "Residual": [227510, 227511, 42837, 43892, 44909, 44959],
    "Pre-admission Output": [40060, 226633],
}

INPUT_ITEMS = {
    "Vasopressin": [30051, 222315],
    "Vancomycin": [225798],
    "Calcium Gluconate": [30023, 221456, 227525, 42504, 43070, 45699, 46591, 44346, 46291],
    "Furosemide": [30123, 221794, 228340],
    "Famotidine": [225907],
    "Piperacillin": [225893, 225892],
    "Cefazolin": [225850],
    "Fiber": [225936, 30166, 30073, 227695, 30088, 225928, 226051, 226050, 226048, 45381, 45597, 227699, 227696, 44218, 45406, 44675, 226049, 44202, 45370, 227698, 226027, 42106, 43994, 45865, 44318, 42091, 44699, 44010, 43134, 44045, 43088, 42641, 45691, 45515, 45777, 42663, 42027, 44425, 45657, 45775, 44631, 44106, 42116, 44061, 44887, 42090, 42831, 45541, 45497, 46789, 44765, 42050],
    "Pantoprazole": [225910, 40549, 41101, 41583, 44008, 40700, 40550],
    "Magnesium Sulphate": [222011, 30027, 227524],
    "KCl": [30026, 225166, 227536],
    "Midazolam": [30124, 221668],
    "Propofol": [30131, 222168],
    "Albumin 25%": [220862, 30009],
    "Albumin 5%": [220864, 30008],
    "Fresh Frozen Plasma": [30005, 220970],
    "Lorazepam": [30141, 221385],
    "Morphine Sulfate": [30126, 225154],
    "Gastric Meds": [30144, 225799],
    "Lactated Ringers": [30021, 225828],
    "Milrinone": [30125, 221986],
    "OR/PACU Crystalloid": [30101, 226364, 30108, 226375],
    "Heparin": [30025, 225975, 225152],
    "Packed RBC": [30001, 225168, 30104, 226368, 227070],
    "PO intake": [30056, 226452, 30109, 226377],
    "Neosynephrine": [30128, 221749, 30127],
    "Piggyback": [226089, 30063],
    "Nitroglycerine": [30121, 222056, 30049],
    "Nitroprusside": [30050, 222051],
    "Metoprolol": [225974],
    "Norepinephrine": [30120, 221906, 30047],
    "Colloid": [30102, 226365, 30107, 226376],
    "Hydralazine": [221828],
    "GT Flush": [226453, 30059],
    "Hydromorphone": [30163, 221833],
    "Fentanyl": [225942, 30118, 221744, 30149],
    "Insulin Regular": [30045, 223258, 30100],
    "Insulin Humalog": [223262],
    "Insulin largine": [223260],
    "Insulin NPH": [223259],
    "Unknown": [30140],
    "D5W": [30013, 220949],
    "Dextrose Other": [30015, 225823, 30060, 225825, 220950, 30016, 30061, 225827, 225941, 30160, 220952, 30159, 30014, 30017, 228142, 228140, 45360, 228141, 41550],
    "Normal Saline": [225158, 30018],
    "Half Normal Saline": [30020, 225159],
    "Sterile Water": [225944, 30065],
    "Free Water": [30058, 225797, 41430, 40872, 41915, 43936, 41619, 42429, 44492, 46169, 42554],
    "Solution": [225943],
    "Dopamine": [30043, 221662],
    "Epinephrine": [30119, 221289, 30044],
    "Amiodarone": [30112, 221347, 228339, 45402],
    "TPN": [30032, 225916, 225917, 30096],
    "Magnesium Sulfate (Bolus)": [227523],
    "KCl (Bolus)": [227522],
    "Pre-admission Intake": [30054, 226361],
}

OUTPUT_LABEL_GROUPS = {
    "Urine": ("urine", "foley", "void", "nephrostomy", "condom", "drainage bag"),
    "Stool": ("stool", "fecal", "colostomy", "ileostomy", "rectal"),
    "Chest Tube": ("chest tube",),
    "Jackson-Pratt": ("jackson",),
}

AUDIT_ONLY_ITEMS = {("chartevents", 224642): "Temperature Site"}


def output_variable(itemid, label):
    for variable, itemids in OUTPUT_FIXED_ITEMS.items():
        if int(itemid) in itemids:
            return variable
    text = "" if label is None else str(label).lower()
    for variable, terms in OUTPUT_LABEL_GROUPS.items():
        if any(term in text for term in terms):
            return variable
    return None


def build_legacy_item_map(d_items):
    mapping = defaultdict(set)
    for variable, itemids in CHART_ITEMS.items():
        for itemid in itemids:
            mapping[("chartevents", int(itemid))].add(variable)
    for variable, itemids in LAB_ITEMS.items():
        for itemid in itemids:
            mapping[("labevents", int(itemid))].add(variable)
            if variable not in {"SO2_bloodgas", "Intubated"}:
                mapping[("chartevents", int(itemid))].add(variable)
    for (table, itemid), variable in AUDIT_ONLY_ITEMS.items():
        mapping[(table, itemid)].add(variable)
    for variable, itemids in INPUT_ITEMS.items():
        for itemid in itemids:
            mapping[("inputevents", int(itemid))].add(variable)
    if d_items is not None and len(d_items):
        for row in d_items.itertuples(index=False):
            variable = output_variable(row.itemid, getattr(row, "label", None))
            if variable:
                mapping[("outputevents", int(row.itemid))].add(variable)
    return dict(mapping)


def legacy_mapping_frame(mapping):
    table_names = {
        "chartevents": "chart", "labevents": "lab", "outputevents": "output",
        "inputevents": "input_mv",
    }
    return pd.DataFrame([
        {"TABLE": table_names[table], "ITEMID": itemid, "NAME": name}
        for (table, itemid), names in mapping.items()
        for name in sorted(names)
    ])
