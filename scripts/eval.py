import sys
import json
from src.infer.decider import Decider

# 간단 평가 스크립트

def main():
    if len(sys.argv) < 2:
        print("데이터 경로가 필요합니다")
        return
    path = sys.argv[1]
    decider = Decider()
    total = 0
    correct = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred = decider.judge(obj["question"])
            if obj.get("gold", "") in pred:
                correct += 1
            total += 1
    if total:
        print(f"accuracy: {correct}/{total}={correct/total:.2f}")
    else:
        print("no data")

if __name__ == "__main__":
    main()
