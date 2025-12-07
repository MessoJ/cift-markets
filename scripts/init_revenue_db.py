import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cift.core.database import DatabaseManager
from cift.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_revenue_db():
    """Initialize the platform_revenue table"""
    settings = get_settings()
    db = DatabaseManager()
    await db.initialize()
    
    try:
        with open('create_revenue_tracking.sql', 'r') as f:
            sql = f.read()
            
        logger.info("Creating platform_revenue table...")
        async with db.pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("Successfully created platform_revenue table and views.")
        
    except Exception as e:
        logger.error(f"Failed to initialize revenue DB: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(init_revenue_db())
