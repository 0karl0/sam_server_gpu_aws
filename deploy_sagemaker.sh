#!/usr/bin/env bash
set -euo pipefail

# Deploy the sam-server2 container to SageMaker and set up shared S3 storage.
# Usage: ./deploy_sagemaker.sh [ecr_image_uri]

IMAGE_URI="${1:-043902793141.dkr.ecr.us-east-1.amazonaws.com/sam-server2:sage}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="${S3_BUCKET:-sam-server-shared-$(date +%s)}"
ROLE_NAME="${SAGEMAKER_ROLE_NAME:-SamServer2SageMakerRole}"
MODEL_NAME="${SAGEMAKER_MODEL_NAME:-sam-server2-model}"
ENDPOINT_CONFIG_NAME="${SAGEMAKER_ENDPOINT_CONFIG_NAME:-sam-server2-endpoint-config}"
ENDPOINT_NAME="${SAGEMAKER_ENDPOINT_NAME:-sam-server2-endpoint}"

# Create S3 bucket for shared storage
aws s3 mb "s3://${BUCKET}" --region "$REGION" 2>/dev/null || echo "Bucket already exists"

# Create or retrieve IAM role for SageMaker
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || \
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"}]}' \
    --query 'Role.Arn' --output text)

aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess >/dev/null
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess >/dev/null

# Register the model
aws sagemaker create-model --region "$REGION" --model-name "$MODEL_NAME" \
  --primary-container Image="$IMAGE_URI",Environment="{SHARED_DIR=/mnt/s3,S3_BUCKET=$BUCKET}" \
  --execution-role-arn "$ROLE_ARN" 2>/dev/null || echo "Model already exists"

# Create endpoint configuration
aws sagemaker create-endpoint-config --region "$REGION" --endpoint-config-name "$ENDPOINT_CONFIG_NAME" \
  --production-variants '[{"VariantName":"AllTraffic","ModelName":"'"$MODEL_NAME"'","InstanceType":"ml.g4dn.xlarge","InitialInstanceCount":1}]' \
  2>/dev/null || echo "Endpoint config already exists"

# Deploy the endpoint
aws sagemaker create-endpoint --region "$REGION" --endpoint-name "$ENDPOINT_NAME" \
  --endpoint-config-name "$ENDPOINT_CONFIG_NAME" 2>/dev/null || echo "Endpoint already exists"

aws sagemaker wait endpoint-in-service --region "$REGION" --endpoint-name "$ENDPOINT_NAME"

echo "S3 bucket: s3://${BUCKET}"
echo "SageMaker endpoint: ${ENDPOINT_NAME}"
echo "Mount the bucket to /mnt/s3 on Server1 and the SageMaker container."
