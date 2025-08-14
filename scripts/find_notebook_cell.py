import json
import sys
from pathlib import Path

def main(nb_path: str, needle: str) -> int:
    p = Path(nb_path)
    with p.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    hits = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src)
        if needle in text:
            hits.append((idx, text))
    for idx, text in hits:
        print(f"INDEX={idx}")
        print("----CELL START----")
        print(text)
        print("----CELL END----")
    if not hits:
        print("NO_MATCH")
        return 1
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/find_notebook_cell.py <notebook.ipynb> <needle>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))


