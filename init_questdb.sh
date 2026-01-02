#!/bin/bash
# Initialize QuestDB tables
curl -G "http://localhost:9000/exec" --data-urlencode "query=CREATE TABLE IF NOT EXISTS trade_executions (timestamp TIMESTAMP, symbol SYMBOL, price DOUBLE, size LONG, side SYMBOL, trade_id STRING, exchange SYMBOL, aggressive_side SYMBOL) TIMESTAMP(timestamp) PARTITION BY DAY;"
