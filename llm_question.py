import json
import re


def llm_question(input_text, llm_model):
    prompt = f"""
다음은 사용자의 입력입니다: "{input_text}"

사용자의 입력에서 다음 정보를 추출해 주세요:
- 방문 시간대 (아침, 점심, 오후, 저녁, 밤 중 하나)
- 맛집 유형 (제주도민 맛집, 관광객 맛집 중 하나)
- 장소 (사용자의 입력에서 사용자의 장소가 명시되면 저장해주세요)

입력된 문장에서 찾을 수 없는 정보가 있다면, 해당 항목을 "None"으로 설정해 주세요.

결과는 아래 JSON 형식으로 출력해 주세요:
{{
  "time": "추출된 시간대",
  "local_choice": "추출된 맛집 유형",
  "location": "추출된 장소"
}}
"""
    response = llm_model.generate_content(prompt)

    if isinstance(response, str):
        response_text = response
    elif hasattr(response, "text"):
        response_text = response.text
    else:
        print("LLM 응답에서 문자열을 추출할 수 없습니다.")
        return None, None, None

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            extracted_data = json.loads(json_text)

            time = extracted_data.get("time", None)
            local_choice = extracted_data.get("local_choice", None)
            location = extracted_data.get("location", None)

            return time, local_choice, location
        except json.JSONDecodeError:
            print("JSON 파싱에 실패했습니다.")
            return None, None, None
    else:
        print("LLM 응답에서 JSON을 찾지 못했습니다.")
        return None, None, None
