import json

def load_json_flexible(path):
    with open(path, "r") as f:
        content = f.read().strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in content.split("\n")
            if line.strip()
        ]