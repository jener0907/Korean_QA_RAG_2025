import sys
from src.infer.decider import Decider

# 한 줄 판단 스크립트

def main():
    if len(sys.argv) < 2:
        print("문장을 입력하세요")
        return
    query = " ".join(sys.argv[1:])
    decider = Decider()
    print(decider.judge(query))

if __name__ == "__main__":
    main()
