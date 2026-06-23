#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Refresh AWS credentials for the `qaihm` profile by re-assuming the workflow's
# OIDC role via STS. Used to avoid the 12-hour AWS session-duration cap during
# long-running CI jobs (e.g. LLM perf collection).
#
# Requires (set by GitHub Actions when the job has `id-token: write`):
#   ACTIONS_ID_TOKEN_REQUEST_URL
#   ACTIONS_ID_TOKEN_REQUEST_TOKEN
#   AWS_ROLE_ARN
#
# Optional:
#   AWS_REGION         (default: us-west-2)
#   AWS_PROFILE_NAME   (default: qaihm)
#   AWS_SESSION_DURATION_SECONDS  (default: 43200)

set -euo pipefail

: "${ACTIONS_ID_TOKEN_REQUEST_URL:?missing — job needs id-token: write permission}"
: "${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?missing — job needs id-token: write permission}"
: "${AWS_ROLE_ARN:?missing — pass secrets.QAIHM_CI_AWS_ROLE_ARN as AWS_ROLE_ARN}"

region="${AWS_REGION:-us-west-2}"
profile="${AWS_PROFILE_NAME:-qaihm}"
duration="${AWS_SESSION_DURATION_SECONDS:-43200}"
session_name="qaihm-llm-perf-$(date +%s)"

if ! command -v aws >/dev/null 2>&1; then
  echo "refresh_aws_creds: aws CLI not on PATH" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "refresh_aws_creds: jq not on PATH" >&2
  exit 1
fi

token_response=$(curl -sSf \
  -H "Authorization: Bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN}" \
  -H "Accept: application/json; api-version=2.0" \
  "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=sts.amazonaws.com")
web_identity_token=$(echo "${token_response}" | jq -r '.value')
if [[ -z "${web_identity_token}" || "${web_identity_token}" == "null" ]]; then
  echo "refresh_aws_creds: failed to fetch OIDC token" >&2
  exit 1
fi

creds=$(aws sts assume-role-with-web-identity \
  --role-arn "${AWS_ROLE_ARN}" \
  --role-session-name "${session_name}" \
  --web-identity-token "${web_identity_token}" \
  --duration-seconds "${duration}" \
  --region "${region}" \
  --output json)

access_key=$(echo "${creds}" | jq -r '.Credentials.AccessKeyId')
secret_key=$(echo "${creds}" | jq -r '.Credentials.SecretAccessKey')
session_token=$(echo "${creds}" | jq -r '.Credentials.SessionToken')
expiration=$(echo "${creds}" | jq -r '.Credentials.Expiration')

for field_name in access_key secret_key session_token; do
  value="${!field_name}"
  if [[ -z "${value}" || "${value}" == "null" ]]; then
    echo "refresh_aws_creds: STS returned empty/null ${field_name}" >&2
    exit 1
  fi
done

aws configure set aws_access_key_id "${access_key}" --profile "${profile}"
aws configure set aws_secret_access_key "${secret_key}" --profile "${profile}"
aws configure set aws_session_token "${session_token}" --profile "${profile}"
aws configure set region "${region}" --profile "${profile}"

echo "refresh_aws_creds: refreshed profile '${profile}'"
echo "refresh_aws_creds:   session_name=${session_name}"
echo "refresh_aws_creds:   expires=${expiration}"

# Verify the profile actually points at the new credentials by calling STS
# back with the qaihm profile. The Arn returned includes our session name, so
# two consecutive refreshes will print two different Arns — easy to confirm
# from CI logs that the refresh is not a no-op.
caller=$(AWS_PROFILE="${profile}" aws sts get-caller-identity --output json)
echo "refresh_aws_creds:   caller_arn=$(echo "${caller}" | jq -r '.Arn')"
