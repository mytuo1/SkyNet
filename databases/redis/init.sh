#!/bin/sh
# Redis initialization — runs once on first start
# Sets up default key namespaces and brain:knobs:current defaults

redis-cli SET brain:initialized "true"
redis-cli SET brain:version "1.0.0"

# Default knobs (all brain services read these)
redis-cli SET brain:knobs:current '{"humor":0.4,"honesty":0.95,"empathy":0.8,"formality":0.4,"verbosity":0.5,"creativity":0.6,"skepticism":0.6,"assertiveness":0.5,"curiosity":0.8,"warmth":0.75,"reasoning_depth":0.7,"analogy_seeking":0.7,"verification_threshold":0.6,"memory_weight":0.6,"novelty_sensitivity":0.7,"swarm_preference":0.4,"framework_threshold":50,"ethical_weight":0.9,"care_explicitness":0.5}'

echo "Redis brain namespace initialized"
