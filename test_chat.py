import requests
import json

# 测试聊天
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "你好", "session_id": "test"}
)
result = response.json()
print(json.dumps(result, ensure_ascii=False, indent=2))

# 测试意图识别
response2 = requests.post(
    "http://localhost:8000/chat",
    json={"query": "我的公司工资政策入职满一年有几天年假，工作日加班有几倍工资？", "session_id": "test"}
)
result2 = response2.json()
print("\n" + "="*50)
print(json.dumps(result2, ensure_ascii=False, indent=2))