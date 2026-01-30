#!/bin/bash

echo "=== System Status ==="
curl -s http://localhost:5000/status | jq -r '.service_info | "Status: \(.status)\nStep: \(.current_step)\nLast Update: \(.last_update)"'

echo -e "\n=== Order Statistics ==="
curl -s http://localhost:5000/stats | jq -r '.status_summary[] | "\(.status): \(.count)"'

echo -e "\n=== 7-Day Execution Rates ==="
curl -s http://localhost:5000/stats | jq -r '.execution_rate_7d[] | "\(.currency)-\(.period)d: \(.exec_rate)% (\(.executed)/\(.total))"'

echo -e "\n=== Validation Tests ==="
curl -s http://localhost:5000/validate | jq -r '.tests | to_entries[] | "\(.key): \(.value.status)"'
