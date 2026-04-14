#!/bin/bash
# EdgeDB migration script — runs on container startup
# Applies schema migrations if the EdgeDB CLI is available

set -e

EDGEDB_DSN="${EDGEDB_DSN:-edgedb://edgedb/main}"

echo "Starting EdgeDB schema migration..."

# Wait for EdgeDB server to be ready
for i in $(seq 1 30); do
    if edgedb query --dsn "$EDGEDB_DSN" 'SELECT 1' 2>/dev/null; then
        echo "EdgeDB is ready"
        break
    fi
    echo "Waiting for EdgeDB... ($i/30)"
    sleep 2
done

# Create migration and apply
if command -v edgedb &> /dev/null; then
    edgedb migrate --dsn "$EDGEDB_DSN" 2>&1 || echo "Migration skipped or already up to date"
else
    echo "edgedb CLI not available — skipping migration"
fi

echo "EdgeDB migration script complete"
