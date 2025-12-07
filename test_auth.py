#!/usr/bin/env python3
"""Test authentication and fix login issues"""

import asyncio
import asyncpg
from passlib.context import CryptContext

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def main():
    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="cift_user", 
        password="cift_password",
        database="cift_markets"
    )
    
    try:
        # Check existing admin user
        admin_user = await conn.fetchrow(
            "SELECT email, username, password_hash FROM users WHERE email = $1",
            "admin@ciftmarkets.com"
        )
        
        if admin_user:
            print(f"Found user: {admin_user['email']} ({admin_user['username']})")
            print(f"Password hash: {admin_user['password_hash'][:20]}...")
            
            # Test password verification
            is_valid = pwd_context.verify("admin", admin_user['password_hash'])
            print(f"Password 'admin' is valid: {is_valid}")
            
            if not is_valid:
                # Update password hash
                new_hash = pwd_context.hash("admin")
                await conn.execute(
                    "UPDATE users SET password_hash = $1 WHERE email = $2",
                    new_hash, "admin@ciftmarkets.com"
                )
                print("✅ Password updated!")
        else:
            # Create admin user
            password_hash = pwd_context.hash("admin")
            await conn.execute("""
                INSERT INTO users (email, username, full_name, password_hash, is_active, is_verified, created_at)
                VALUES ($1, $2, $3, $4, true, true, NOW())
                ON CONFLICT (email) DO UPDATE SET 
                password_hash = $4, is_active = true, is_verified = true
            """, "admin@ciftmarkets.com", "admin", "Administrator", password_hash)
            print("✅ Admin user created!")
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
