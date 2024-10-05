import json
import os
import re

import faiss
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv

from llm_question import llm_question

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
search_api_key = os.getenv("SEARCH_API_KEY")
cx = os.getenv("CX")

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "./data")
module_dir = os.path.join(base_dir, "./modules")
csv_file_dir = os.path.join(data_dir, "JEJU_MCT_DATA_v2.csv")
faiss_index_path = os.path.join(module_dir, "faiss_index.index")


genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel("gemini-1.5-flash")


df = pd.read_csv(csv_file_dir, encoding="cp949")
df.columns = df.columns.str.strip()


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


existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
existing_categorical_columns = [col for col in categorical_columns if col in df.columns]


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
    df[col] = df[col].apply(map_category_to_numeric)

all_feature_columns = existing_numeric_columns + existing_categorical_columns


features = df[all_feature_columns].fillna(0).astype("float32")


feature_vectors = features.to_numpy()


def load_faiss_index(index_path):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")


faiss_index = load_faiss_index(faiss_index_path)


def generate_numerical_features_from_query(question, time, local_choice):

    prompt = f"""
아래는 사용자의 질문과 선호도입니다:

사용자 질문: "{question}"
방문 시간대: "{time}"
맛집 유형: "{local_choice}"

위 정보를 바탕으로 다음의 특성에 대한 값을 0에서 1 사이의 숫자로 추정하여 JSON 형식으로 제공해 주세요.
특성 목록:
{", ".join(all_feature_columns)}

응답은 **아무 설명 없이** 아래 형식의 JSON 데이터만 출력해 주세요:

{{
    "MON_UE_CNT_RAT": 0.1,
    "TUE_UE_CNT_RAT": 0.2,
    ...
}}

주의사항:
- 응답에는 JSON 데이터만 포함되어야 합니다.
- 추가적인 텍스트나 설명을 포함하지 마세요.
"""
    response = llm_model.generate_content(prompt)

    if isinstance(response, str):
        response_text = response
    else:
        response_text = response.text

    # print("LLM 응답:", response_text)

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            feature_values = json.loads(json_text)

            feature_vector = []
            for feature in all_feature_columns:
                value = feature_values.get(feature, 0)
                feature_vector.append(float(value))
            return np.array(feature_vector, dtype="float32")
        except json.JSONDecodeError:
            print("JSON 파싱에 실패했습니다.")
            return None
    else:
        print("응답에서 JSON을 찾지 못했습니다.")
        return None


def search_google(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": search_api_key,
        "cx": cx,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Google 검색 실패: {response.status_code}")
        return None


def get_search_results(search_response):
    if not search_response or "items" not in search_response:
        print("검색 결과가 없습니다.")
        return []

    results = []
    for item in search_response.get("items", []):
        title = item.get("title", "제목 없음")
        link = item.get("link", "링크 없음")
        snippet = item.get("snippet", "설명 없음")

        results.append({"title": title, "link": link, "snippet": snippet})
    return results


def generate_recommendation_with_llm(question, df, time, local_choice, location, k=5):

    query_vector = generate_numerical_features_from_query(question, time, local_choice)
    if query_vector is None:
        return "질문을 이해하지 못했습니다. 다시 시도해 주세요."

    query_vector = query_vector.reshape(1, -1)
    if query_vector.shape[1] != faiss_index.d:
        print("쿼리 벡터의 차원이 FAISS 인덱스와 일치하지 않습니다.")
        return "추천을 생성하는데 오류가 발생했습니다."

    distances, indices = faiss_index.search(query_vector, k)

    filtered_df = df.iloc[indices[0, :]].reset_index(drop=True)
    if filtered_df.empty:
        return "조건에 맞는 추천 결과가 없습니다."
    store_names = []
    reference_info = ""
    for idx, row in filtered_df.iterrows():
        reference_info += f"{row['MCT_NM']} ({row['MCT_TYPE']}, {row['ADDR']})\n"
        store_names.append(row["MCT_NM"])

    prompt = f"""
사용자 질문: "{question}"
추천할 가맹점 정보:
{reference_info}
위 가맹점을 추천하는 이유와 함께 사용자에게 답변해 주세요.
"""
    response = llm_model.generate_content(prompt)
    search_results_text = ""

    for store_name in store_names:
        search_response = search_google(store_name)
        search_results = get_search_results(search_response)

        search_results_text += f"\n'{store_name}'에 대한 추가 정보 검색 결과:\n"
        if not search_results:
            search_results_text += "추가 정보를 찾을 수 없습니다.\n"
        else:
            for idx, result in enumerate(search_results, start=1):
                search_results_text += f"--- 결과 {idx} ---\n"
                search_results_text += f"제목: {result['title']}\n"
                search_results_text += f"링크: {result['link']}\n"
                search_results_text += f"설명: {result['snippet']}\n\n"

    combined_response = f"{response}\n\n--- 추가 정보 ---\n{search_results_text}"
    return response if isinstance(response, str) else response.text


def main():
    while True:
        print("\n--- 제주 맛집 추천 시스템 ---")
        print("1. 추천형 질문")
        print("2. 종료")
        choice = input("원하는 작업을 선택하세요 (1/2): ")

        if choice == "1":
            input_text = input(
                "추천형 질문을 입력하세요 (예: '제주도에서 아침에 갈만한 관광객 맛집 추천해줘'): "
            )

            time, local_choice, location = llm_question(input_text, llm_model)

            if not time or not local_choice:
                print("시간대나 맛집 유형을 인식하지 못했습니다. 다시 시도해 주세요.")
                continue

            response = generate_recommendation_with_llm(
                input_text, df, time, local_choice, location
            )
            print(f"추천 결과:\n{response}")

        elif choice == "2":
            print("프로그램을 종료합니다.")
            break

        else:
            print("잘못된 입력입니다. 다시 시도하세요.")


if __name__ == "__main__":
    main()
