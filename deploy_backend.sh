#!/bin/bash
set -e

# --- CONFIGURATION ---
# REPLACE THESE VALUES WITH YOUR AWS DETAILS
AWS_ACCOUNT_ID="558101068643"
AWS_REGION="eu-central-1"
ECR_REPO_NAME="size-estimator"
LAMBDA_FUNCTION_NAME="size_estimator_function"

# Image URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest"

echo "1. Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "2. Building Docker Image (Platform: linux/amd64 for Lambda)..."
# Using linux/amd64 is critical for AWS Lambda
# Using --no-cache to ensure updated requirements are picked up
docker build --no-cache --platform linux/amd64 -f Dockerfile.aws -t $ECR_REPO_NAME .

echo "3. Tagging Image..."
docker tag $ECR_REPO_NAME:latest $ECR_URI

echo "4. Pushing Image to ECR..."
docker push $ECR_URI

echo "5. Updating Lambda Function Code..."
aws lambda update-function-code \
    --function-name $LAMBDA_FUNCTION_NAME \
    --image-uri $ECR_URI \
    --region $AWS_REGION

echo "Waiting for code update to complete..."
aws lambda wait function-updated --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION

echo "6. Updating Lambda Configuration (Memory: 3008MB, Timeout: 300s, Storage: 3GB)..."
aws lambda update-function-configuration \
    --function-name $LAMBDA_FUNCTION_NAME \
    --memory-size 3008 \
    --timeout 300 \
    --ephemeral-storage '{"Size": 3072}' \
    --environment "Variables={MODEL_BUCKET=size-estimator-models-${AWS_ACCOUNT_ID}}" \
    --region $AWS_REGION

echo "Done! Backend deployed."
