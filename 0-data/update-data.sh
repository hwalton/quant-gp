#!/bin/bash

OUTPUT_FILE="btc_weekly_prices.csv"
API_URL="https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
VS_CURRENCY="usd"
DAYS=365

echo "Fetching Bitcoin price data from the past ${DAYS} days..."

# Fetch data
response=$(curl -s -w "\n%{http_code}" "${API_URL}?vs_currency=${VS_CURRENCY}&days=${DAYS}")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d') # remove last line (status code)

# Check HTTP status
if [[ "$http_code" -ne 200 ]]; then
    echo "Error: HTTP $http_code while fetching data"
    echo "Response body:"
    echo "$body"
    exit 1
fi

# Validate response contains prices
if ! echo "$body" | jq -e '.prices' > /dev/null; then
    echo "Error: API response does not contain '.prices'"
    echo "Full response:"
    echo "$body"
    exit 1
fi

echo "Processing data to weekly samples..."
echo "date,price_usd" > "$OUTPUT_FILE"

# Extract timestamp/price pairs, reduce to first per ISO week
echo "$body" | jq -r '.prices[][]' | paste - - | while read -r timestamp price; do
    # Convert to ISO week format: YYYY-Www
    week=$(date -d @"$((timestamp / 1000))" +%G-%V)
    # Also capture a representative date (e.g. Monday of that week)
    date=$(date -d @"$((timestamp / 1000))" +%Y-%m-%d)
    echo "$week,$date,$price"
done | awk -F, '!seen[$1]++ { print $2 "," $3 }' >> "$OUTPUT_FILE"

echo "Done. Weekly Bitcoin prices saved to $OUTPUT_FILE"
