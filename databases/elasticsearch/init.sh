#!/bin/bash
# Elasticsearch initialization script
# Waits for ES to be ready, then performs basic setup

set -e

ES_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"

echo "Waiting for Elasticsearch to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${ES_URL}/_cluster/health" > /dev/null 2>&1; then
        echo "Elasticsearch is ready"
        break
    fi
    echo "Waiting for Elasticsearch... ($i/30)"
    sleep 2
done

# Verify cluster health
HEALTH=$(curl -sf "${ES_URL}/_cluster/health" 2>/dev/null | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
echo "Cluster health: ${HEALTH}"

echo "Elasticsearch initialization complete"
