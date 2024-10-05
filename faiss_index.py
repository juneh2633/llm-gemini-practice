import os

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "./data")
module_dir = os.path.join(base_dir, "./modules")
csv_file_dir = os.path.join(data_dir, "JEJU_MCT_DATA_v2.csv")
faiss_index_path = os.path.join(module_dir, "faiss_index.index")
data = pd.read_csv(csv_file_dir, encoding="cp949")
data.columns = data.columns.str.strip()

numeric_columns = [
    "MON_UE_CNT_RAT",
    "TUE_UE_CNT_RAT",
    "WED_UE_CNT_RAT",
    "THU_UE_CNT_RAT",
    "FRI_UE_CNT_RAT",
    "SAT_UE_CNT_RAT",
    "SUN_UE_CNT_RAT",
    "HR_5_11_UE_CNT_RAT",
    "HR_12_13_UE_CNT_RAT",
    "HR_14_17_UE_CNT_RAT",
    "HR_18_22_UE_CNT_RAT",
    "HR_23_4_UE_CNT_RAT",
    "LOCAL_UE_CNT_RAT",
    "RC_M12_MAL_CUS_CNT_RAT",
    "RC_M12_FME_CUS_CNT_RAT",
    "RC_M12_AGE_UND_20_CUS_CNT_RAT",
    "RC_M12_AGE_30_CUS_CNT_RAT",
    "RC_M12_AGE_40_CUS_CNT_RAT",
    "RC_M12_AGE_50_CUS_CNT_RAT",
    "RC_M12_AGE_OVR_60_CUS_CNT_RAT",
]


categorical_columns = ["UE_CNT_GRP", "UE_AMT_GRP", "UE_AMT_PER_TRSN_GRP"]

existing_numeric_columns = [col for col in numeric_columns if col in data.columns]
existing_categorical_columns = [
    col for col in categorical_columns if col in data.columns
]


category_mapping = {
    "1_상위10%이하": 1,
    "2_10~25%": 2,
    "3_25~50%": 3,
    "4_50~75%": 4,
    "5_75~90%": 5,
    "6_90% 초과(하위 10% 이하)": 6,
}


def map_category_to_numeric(s):
    s = s.strip()
    return category_mapping.get(s, np.nan)


for col in existing_categorical_columns:
    data[col] = data[col].apply(map_category_to_numeric)

all_feature_columns = existing_numeric_columns + existing_categorical_columns


features = data[all_feature_columns].fillna(0).astype("float32")


feature_vectors = features.to_numpy()


d = feature_vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(feature_vectors)

faiss.write_index(index, "faiss_index.index")


query_vector = feature_vectors[0].reshape(1, -1)
k = 5
distances, indices = index.search(query_vector, k)

print("Distances:", distances)
print("Indices:", indices)
