from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print(sorted([p.name for p in data_dir.glob("*.pdf")]))