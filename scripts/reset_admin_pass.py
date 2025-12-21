#!/usr/bin/env python3
"""Reset admin password."""
import asyncio

import asyncpg
import bcrypt


async def main():
    conn = await asyncpg.connect(
        host='postgres',
        port=5432,
        user='cift_user',
        password='changeme123',
        database='cift_markets'
    )

    password = 'AdminPass123!'
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt).decode('utf-8')

    await conn.execute(
        'UPDATE users SET hashed_password = $1 WHERE email = $2',
        hashed,
        'admin@ciftmarkets.com'
    )

    await conn.close()
    print('Password reset to AdminPass123!')

asyncio.run(main())
