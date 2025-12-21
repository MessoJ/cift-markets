#!/usr/bin/env python3
"""Fix authentication password hash"""

import asyncio

import asyncpg
import bcrypt


async def fix_admin_password():
    """Fix the admin user password"""
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="cift_user",
            password="cift_password",
            database="cift_markets"
        )

        # Create a proper bcrypt hash
        password = "admin"
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        hash_str = hashed.decode('utf-8')

        print(f"Generated hash: {hash_str}")

        # Update user in database
        result = await conn.execute(
            "UPDATE users SET hashed_password = $1 WHERE email = $2",
            hash_str,
            "admin@ciftmarkets.com"
        )

        print(f"Updated: {result}")

        # Verify the hash works
        stored_user = await conn.fetchrow(
            "SELECT email, hashed_password FROM users WHERE email = $1",
            "admin@ciftmarkets.com"
        )

        if stored_user:
            stored_hash = stored_user['hashed_password'].encode('utf-8')
            is_valid = bcrypt.checkpw(password_bytes, stored_hash)
            print(f"Password verification: {is_valid}")
            print("✅ Admin password fixed!")

        await conn.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(fix_admin_password())
