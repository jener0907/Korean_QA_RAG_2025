import re
import uuid
import hashlib
from typing import List, Dict

# 규정 텍스트를 청크로 분할

def split_rules(text: str) -> List[Dict]:
    result = []
    pattern = re.compile(r"(제\d+항)")
    parts = pattern.split(text)
    if not parts:
        return result
    current_rule = None
    for part in parts:
        if pattern.match(part):
            current_rule = part
            continue
        if not current_rule:
            continue
        body = part.strip()
        if not body:
            continue
        # 400-800자 보조 분할
        for i in range(0, len(body), 800):
            chunk = body[i:i+800]
            examples = []
            exceptions = []
            for line in chunk.split('\n'):
                if line.startswith("예외") or line.startswith("주의"):
                    exceptions.append(line)
                if line.startswith("용례") or line.startswith("예문"):
                    examples.append(line)
            doc = {
                "id": str(uuid.uuid4()),
                "rule_id": current_rule,
                "topic": "",
                "body": chunk,
                "examples": examples,
                "exceptions": exceptions,
                "source": {"name": "국립국어원", "version": "", "url": ""},
            }
            doc["hash"] = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            result.append(doc)
    return result
