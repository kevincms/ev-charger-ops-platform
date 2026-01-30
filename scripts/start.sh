#!/bin/bash

set -euo pipefail

APP_DIR=/home/ubuntu/app
JAR_PATH=$(find $APP_DIR -name "*.jar" | grep build/libs | head -n 1)

if [ -z "$JAR_PATH" ]; then
  echo "JAR not found" >> $APP_DIR/app.log
  exit 1
fi

MYSQL_URL=$(aws ssm get-parameter \
  --name "/MYSQL/MYSQL_URL" \
  --query "Parameter.Value" \
  --output text)

MYSQL_USERNAME=$(aws ssm get-parameter \
  --name "/MYSQL/MYSQL_USERNAME" \
  --query "Parameter.Value" \
  --output text)

MYSQL_PASSWORD=$(aws ssm get-parameter \
  --name "/MYSQL/MYSQL_PASSWORD" \
  --query "Parameter.Value" \
  --output text)

AWS_S3_BUCKET=$(aws ssm get-parameter \
  --name "/S3-BUCKET-Name" \
  --query "Parameter.Value" \
  --output text)

export MYSQL_URL
export MYSQL_USERNAME
export MYSQL_PASSWORD
export AWS_S3_BUCKET

echo "Starting $JAR_PATH" >> $APP_DIR/app.log

nohup java -jar "$JAR_PATH" --spring.profiles.active=prod \
  > "$APP_DIR/app.log" 2>&1 &

echo $! > $APP_DIR/app.pid
