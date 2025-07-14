#!/bin/bash

# Output CSV path
OUTPUT_FILE="btc_monthly_prices.csv"

# Yahoo Finance hidden endpoint for BTC monthly data
CSV_URL="https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1410912000&period2=1752518271&interval=1mo&events=history&includeAdjustedClose=true"

echo "Downloading BTC monthly price data from Yahoo Finance..."
curl -s -o "$OUTPUT_FILE" "$CSV_URL"

# Check if download was successful
if [[ $? -ne 0 || ! -s "$OUTPUT_FILE" ]]; then
    echo "Error: Failed to download or save the CSV data."
    exit 1
fi

echo "Original CSV saved to $OUTPUT_FILE"

# ─── Optional: Format to match 'date,price_usd' format ──────────────────────────

FORMATTED_FILE="btc_monthly_prices_formatted.csv"
echo "date,price_usd" > "$FORMATTED_FILE"

tail -n +2 "$OUTPUT_FILE" | awk -F',' '{ printf "%s,%.6f\n", $1, $5 }' >> "$FORMATTED_FILE"

echo "Formatted monthly price data saved to $FORMATTED_FILE"
