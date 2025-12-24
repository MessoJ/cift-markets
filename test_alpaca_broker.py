#!/usr/bin/env python3
"""Test Alpaca Broker API connection."""
import requests
import base64
import json

api_key = "CKRWFWVDTAAWPDI2WOV353CAIB"
secret_key = "A9y8HrGVrMnJoCt2mQk4mND5UMHmiWd18HZ6qrAT1iUm"

credentials = base64.b64encode(f"{api_key}:{secret_key}".encode()).decode()
headers = {"Authorization": f"Basic {credentials}"}

# Try sandbox first
base_url = "https://broker-api.sandbox.alpaca.markets"
print(f"Testing Broker API at {base_url}...")

# List accounts
response = requests.get(f"{base_url}/v1/accounts", headers=headers)
print(f"\n1. List Accounts - Status: {response.status_code}")
if response.status_code == 200:
    accounts = response.json()
    print(f"   Found {len(accounts)} accounts:")
    for acc in accounts[:5]:  # Show first 5
        print(f"   - ID: {acc.get('id')}, Status: {acc.get('status')}, Created: {acc.get('created_at')}")
        
        # Get Trading Account Details
        trade_resp = requests.get(f"{base_url}/v1/trading/accounts/{acc.get('id')}/account", headers=headers)
        if trade_resp.status_code == 200:
            trade_data = trade_resp.json()
            print(f"     💰 Cash: ${trade_data.get('cash')}")
            print(f"     💪 Buying Power: ${trade_data.get('buying_power')}")
            print(f"     📈 Equity: ${trade_data.get('equity')}")
            
            # Fund account if empty (Sandbox only)
            if float(trade_data.get('cash', 0)) < 1000:
                print("     Funding account with $100,000...")
                
                # 1. Check for existing ACH Relationship
                rel_id = None
                ach_list_resp = requests.get(f"{base_url}/v1/accounts/{acc.get('id')}/ach_relationships", headers=headers)
                if ach_list_resp.status_code == 200 and len(ach_list_resp.json()) > 0:
                    rel_id = ach_list_resp.json()[0].get('id')
                    print(f"     ✅ Found existing ACH Relationship: {rel_id}")
                else:
                    # Create new if none
                    ach_data = {
                        "account_owner_name": "CIFT Markets",
                        "bank_account_type": "CHECKING",
                        "bank_account_number": "000123456789",
                        "bank_routing_number": "110000000",
                        "nickname": "Test Bank"
                    }
                    ach_resp = requests.post(f"{base_url}/v1/accounts/{acc.get('id')}/ach_relationships", headers=headers, json=ach_data)
                    if ach_resp.status_code == 200:
                        rel_id = ach_resp.json().get('id')
                        print(f"     ✅ Created ACH Relationship: {rel_id}")
                    else:
                        print(f"     ❌ ACH Creation Failed: {ach_resp.text}")

                # 2. Initiate Transfer
                if rel_id:
                    fund_data = {
                        "amount": "50000",
                        "direction": "INCOMING",
                        "transfer_type": "ach",
                        "relationship_id": rel_id
                    }
                    fund_resp = requests.post(f"{base_url}/v1/accounts/{acc.get('id')}/transfers", headers=headers, json=fund_data)
                    print(f"     Funding Status: {fund_resp.status_code}")
                    if fund_resp.status_code == 200:
                        print(f"     ✅ Funding Initiated: {fund_resp.json().get('id')}")
                    else:
                        print(f"     ❌ Funding Failed: {fund_resp.text}")

        else:
            print(f"     ❌ Failed to get trading details: {trade_resp.status_code}")

# Try production API
base_url_prod = "https://broker-api.alpaca.markets"
print(f"\nTesting Production Broker API at {base_url_prod}...")
response = requests.get(f"{base_url_prod}/v1/accounts", headers=headers)
print(f"2. List Accounts (Production) - Status: {response.status_code}")
if response.status_code == 200:
    accounts = response.json()
    print(f"   Found {len(accounts)} accounts:")
    for acc in accounts[:5]:
        print(f"   - ID: {acc.get('id')}, Status: {acc.get('status')}")
else:
    print(f"   Error: {response.text[:500]}")
