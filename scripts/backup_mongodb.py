"""
Backup all MongoDB collections to local CSV files.
Run before any schema changes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from bson import ObjectId
from config.db import get_db, DB_NAME

BACKUP_DIR = Path(__file__).parent.parent / "backups"


def _sanitize(doc: dict) -> dict:
    """Convert BSON types that pandas/csv can't handle."""
    clean = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            clean[k] = str(v)
        elif isinstance(v, bytes):
            clean[k] = f"<binary {len(v)} bytes>"
        elif hasattr(v, 'item'):
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


def backup_collection(name: str, timestamp: str) -> int:
    db   = get_db()
    docs = list(db[name].find({}))
    if not docs:
        print(f"  {name}: empty — skipped")
        return 0

    rows = [_sanitize(d) for d in docs]
    df   = pd.DataFrame(rows)

    out  = BACKUP_DIR / f"{name}_{timestamp}.csv"
    df.to_csv(out, index=False)
    print(f"  {name}: {len(df)} rows saved to {out}")
    return len(df)


def main():
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Backing up MongoDB db='{DB_NAME}' at {ts}\n")

    db          = get_db()
    collections = db.list_collection_names()
    print(f"Collections found: {collections}\n")

    total = 0
    for col in collections:
        total += backup_collection(col, ts)

    print(f"\nDone. {total} total documents backed up to {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
