#!/usr/bin/env bash
set -e

[[ -z "$1" ]] && { echo "ERROR: RDK commit SHA required as first argument" >&2; exit 1; }
[[ -z "$MEDIA_USER_TOKEN" ]] && { echo "ERROR: MEDIA_USER_TOKEN required" >&2; exit 1; }

RIVERMAX_BASE_URL="https://api.github.com/repos/Mellanox/rivermax"
POLL_INTERVAL_SEC=300   # 5 minutes between poll cycles
API_SLEEP_SEC=5         # sleep between tree API calls to avoid rate limits
STATUS_INTERVAL_SEC=900   # 15 minutes between CI status checks
SUBMODULE_PATH="dev-kit"
RDK_SHA="$1"

api_request() {
  local url="$1" output code body

  if ! output=$(curl -s -S --connect-timeout 30 --max-time 120 -w "\n%{http_code}" \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: Bearer $MEDIA_USER_TOKEN" \
    "$url"); then
    echo "ERROR: curl failed for $url" >&2
    return 1
  fi

  code="${output##*$'\n'}"
  body="${output%$'\n'*}"

  if ! [[ "$code" =~ ^[0-9]{3}$ ]]; then
    echo "ERROR: could not parse curl response for $url" >&2
    return 1
  fi
  if [[ "$code" -ge 400 ]]; then
    echo "ERROR: API request failed (HTTP $code) for $url: $body" >&2
    return 1
  fi

  echo "$body"
}

# Returns 0 if PR pointing at RDK_SHA.
submodule_matches_sha() {
  local pr_head_sha="$1" tree
  if ! tree=$(api_request "${RIVERMAX_BASE_URL}/git/trees/${pr_head_sha}"); then
    echo "ERROR: could not fetch git tree for head ${pr_head_sha:0:7}." >&2
    return 1
  fi
  jq -e \
    --arg p "$SUBMODULE_PATH" \
    --arg s "$RDK_SHA" \
    'any(.tree[]?; .path == $p and (.mode | tostring) == "160000" and .sha == $s)' <<<"$tree" >/dev/null || return 1
}

# Scans one page of PRs JSON for PR pointing at RDK_SHA.
scan_open_prs_page() {
  local response="$1" pr_count="$2" i pr pr_head_sha pr_num

  for (( i=0; i<pr_count; i++ )); do
    if ! pr=$(jq -c ".[$i]" <<<"$response") || ! pr_head_sha=$(jq -r '.head.sha // empty' <<<"$pr"); then
      echo "ERROR: PR list entry at index ${i} is invalid." >&2
      sleep "$API_SLEEP_SEC"
      continue
    fi
    if submodule_matches_sha "$pr_head_sha"; then
      RIVERMAX_PR_SHA="$pr_head_sha"
      pr_num=$(jq -r '.number // empty' <<<"$pr") || true
      if [[ -n "$pr_num" ]]; then
        RIVERMAX_PR_LABEL="PR #${pr_num}"
      else
        RIVERMAX_PR_LABEL="PR head commit ${pr_head_sha:0:7}"
      fi
      echo "Found matching ${RIVERMAX_PR_LABEL}: $(jq -r '.html_url // empty' <<<"$pr" || true)"
      return 0
    fi
    sleep "$API_SLEEP_SEC"
  done
  return 1
}

# Finds an open Rivermax PR pointing at RDK_SHA.
# Sets RIVERMAX_PR_SHA and RIVERMAX_PR_LABEL on success. Loops until found.
find_matching_rivermax_pr() {
  local page=1 response pr_count

  echo "Searching for an open Rivermax PR with dev-kit submodule at ${RDK_SHA:0:7}..."

  while true; do
    # List open PRs page by page
    if ! response=$(api_request "${RIVERMAX_BASE_URL}/pulls?state=open&per_page=100&page=${page}") ||
       ! pr_count=$(jq -r 'if type == "array" then length else 0 end' <<<"$response") ||
       ! [[ "$pr_count" =~ ^[0-9]+$ ]]; then
      echo "ERROR: could not list open PRs. Retrying in ${POLL_INTERVAL_SEC}s..."
      sleep "$POLL_INTERVAL_SEC"
      continue
    fi

    if scan_open_prs_page "$response" "$pr_count"; then
      return 0
    fi

    if [[ "$pr_count" -lt 100 ]]; then
      page=1
      echo "Full scan done. Waiting ${POLL_INTERVAL_SEC}s before next cycle..."
      sleep "$POLL_INTERVAL_SEC"
    else
      echo "Fetching next PR list page in ${API_SLEEP_SEC}s..."
      sleep "$API_SLEEP_SEC"
      page=$((page + 1))
    fi
  done
}

# Polls commit status until success or failure.
wait_for_ci_status() {
  local status_response state

  echo "Waiting for Rivermax CI for ${RIVERMAX_PR_LABEL}..."

  while true; do
    # Check Rivermax CI status
    if ! status_response=$(api_request "${RIVERMAX_BASE_URL}/commits/${RIVERMAX_PR_SHA}/status") ||
       ! state=$(jq -r '.state // ""' <<<"$status_response"); then
      echo "ERROR: could not check CI status. Retrying in ${STATUS_INTERVAL_SEC}s..."
      sleep "$STATUS_INTERVAL_SEC"
      continue
    fi
    case "$state" in
      success)
        echo "Rivermax CI for ${RIVERMAX_PR_LABEL} passed"
        exit 0
        ;;
      failure|error)
        echo "ERROR: Rivermax CI for ${RIVERMAX_PR_LABEL} failed - check Rivermax CI logs for details" >&2
        exit 1
        ;;
      pending|"")
        echo "Rivermax CI still pending. Checking again in ${STATUS_INTERVAL_SEC}s..."
        ;;
      *)
        echo "ERROR: Unexpected CI state: $state" >&2
        exit 1
        ;;
    esac

    sleep "$STATUS_INTERVAL_SEC"
  done
}

find_matching_rivermax_pr
wait_for_ci_status
