import asyncio
import os
import sys

sys.path.append(os.getcwd())
import asyncpg

from cift.core.config import settings


async def check_users():
    try:
        conn = await asyncpg.connect(settings.postgres_url)

        print("\n--- Users ---")
        users = await conn.fetch("SELECT id, email, full_name, is_superuser FROM users")
        for u in users:
            print(f"ID: {u['id']}, Email: {u['email']}, Name: {u['full_name']}, Superuser: {u['is_superuser']}")

            # Check snapshots count for this user
            count = await conn.fetchval("SELECT COUNT(*) FROM portfolio_snapshots WHERE user_id = $1", u['id'])
            print(f"  -> Snapshots: {count}")

        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_users())
