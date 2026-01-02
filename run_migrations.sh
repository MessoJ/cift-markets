#!/bin/bash
set -e

echo "Waiting for Postgres to be ready..."
until sudo docker exec cift-postgres pg_isready -U cift_user -d cift_markets; do
  echo "Postgres is unavailable - sleeping"
  sleep 1
done

echo "Running migrations..."
for file in $(ls cift-markets/database/migrations/*.sql | sort); do
  echo "Running $file..."
  cat "$file" | sudo docker exec -i cift-postgres psql -U cift_user -d cift_markets
done

echo "Migrations complete!"
