#!/usr/bin/env bash
set -euo pipefail

# Simple script to invoke a SageMaker endpoint.
# Usage: ./trigger_sagemaker.sh [endpoint_name] [input_json] [output_file]
# Defaults: endpoint_name="sam-server2-endpoint", input_json="input.json", output_file="output.json"

ENDPOINT_NAME="${1:-sam-server2-endpoint}"
INPUT_FILE="${2:-input.json}"
OUTPUT_FILE="${3:-output.json}"

aws sagemaker-runtime invoke-endpoint \
  --endpoint-name "$ENDPOINT_NAME" \
  --body fileb://"$INPUT_FILE" \
  --content-type application/json \
  "$OUTPUT_FILE"

cat "$OUTPUT_FILE"
