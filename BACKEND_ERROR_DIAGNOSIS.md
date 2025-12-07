# Backend Error Diagnosis

## Issue
Backend is running but returning 500 errors for:
- `/api/v1/globe/ships?min_importance=0`
- `/api/v1/globe/countries/{code}?timeframe=24h`

## Root Cause
The `ships_current_status` table likely **doesn't exist** in the database.

The globe.py endpoint queries:
```sql
FROM ships_current_status
WHERE is_active = true
```

But this table was never created in the database schema.

## Solution Options

### Option 1: Return Empty Ships List (Quick Fix)
Wrap the query in try/catch and return empty list if table doesn't exist.

### Option 2: Use asset_locations for Ships (If ships were seeded there)
Check if ships are in `asset_locations` table with `asset_type = 'ship'`.

### Option 3: Create Ships Table Schema
Create the proper `ships_current_status` table with migration.

## Quick Fix Implementation
Modify the ships endpoint to handle missing table gracefully.
