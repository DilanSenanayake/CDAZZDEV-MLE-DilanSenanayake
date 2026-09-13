from pathlib import Path

root = Path(r"d:\Github\CDAZZDEV assignment")
for p in root.rglob("*"):
    if p.suffix.lower() not in {".py", ".md", ".txt", ".json", ".jsonl", ".example"}:
        continue
    if ".git" in p.parts or "__pycache__" in p.parts:
        continue
    raw = p.read_bytes()
    try:
        raw.decode("utf-8")
        continue
    except UnicodeDecodeError:
        pass
    text = raw.decode("latin-1")
    for a, b in [
        ("\x97", "-"),
        ("\x96", "-"),
        ("\xb7", "-"),
        ("\x91", "'"),
        ("\x92", "'"),
        ("\x93", '"'),
        ("\x94", '"'),
        ("\x9d", ""),
    ]:
        text = text.replace(a, b)
    p.write_text(text, encoding="utf-8")
    print("fixed", p)
