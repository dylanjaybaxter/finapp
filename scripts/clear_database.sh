#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
DB_PATH="$PROJECT_ROOT/data/finance.db"

if [[ ! -f "$DB_PATH" ]]; then
    echo "No database found at $DB_PATH"
    exit 0
fi

echo "This will delete the database: $DB_PATH"
read -r -p "Type 'yes' to confirm: " CONFIRM || true
if [[ "${CONFIRM:-}" != "yes" ]]; then
    echo "Aborted."
    exit 1
fi

rm -f "$DB_PATH"
echo "Deleted $DB_PATH"

# Recreate empty schema
python - <<'PY'
import os
from finance_dashboard import db

db.init_db()
print("Recreated empty database schema.")
PY
