#!/bin/bash
# EdgeDB migration script — runs on container startup
# Applies schema migrations if the EdgeDB CLI is available

set -e

EDGEDB_DSN="${EDGEDB_DSN:-edgedb://edgedb/main}"

echo "Starting EdgeDB schema migration..."

# Wait for EdgeDB server to be ready (connect to default database)
for i in $(seq 1 30); do
    if edgedb query --dsn "edgedb://localhost:5656" --tls-security=insecure 'SELECT 1' 2>/dev/null; then
        echo "EdgeDB is ready"
        break
    fi
    echo "Waiting for EdgeDB... ($i/30)"
    sleep 2
done

# Create the 'main' database if it doesn't exist
edgedb query --dsn "edgedb://localhost:5656" --tls-security=insecure \
  "CREATE DATABASE main" 2>&1 || echo "Database 'main' creation attempted (may already exist)"

# Create migration and apply
if command -v edgedb &> /dev/null; then
    edgedb migrate --dsn "$EDGEDB_DSN" --tls-security=insecure 2>&1 || echo "Migration skipped or already up to date"
else
    echo "edgedb CLI not available — skipping migration"
fi

echo "EdgeDB migration script complete"
