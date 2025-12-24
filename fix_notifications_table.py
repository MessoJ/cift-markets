import asyncio
import os
import sys
from uuid import uuid4

# Add project root to path
sys.path.append(os.getcwd())

from cift.core.database import get_postgres_pool
from loguru import logger

async def fix_notifications_table():
    logger.info("Checking notifications table...")
    
    try:
        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            # Check if table exists
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'notifications'
                );
            """)
            
            if exists:
                logger.info("✅ Notifications table already exists.")
                
                # Check columns
                columns = await conn.fetch("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'notifications';
                """)
                col_names = [r['column_name'] for r in columns]
                logger.info(f"Columns: {col_names}")
                
                return

            logger.info("⚠️ Notifications table missing. Creating it...")
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL,
                    notification_type VARCHAR(50) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    read_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB,
                    
                    CONSTRAINT fk_user
                        FOREIGN KEY(user_id) 
                        REFERENCES users(id)
                        ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
                CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at DESC);
            """)
            
            logger.info("✅ Notifications table created successfully.")
            
    except Exception as e:
        logger.error(f"❌ Failed to fix notifications table: {e}")

if __name__ == "__main__":
    asyncio.run(fix_notifications_table())
