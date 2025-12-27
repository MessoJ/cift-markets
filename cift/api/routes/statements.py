"""
ACCOUNT STATEMENTS & TAX DOCUMENTS API ROUTES
Handles account statements, tax forms (1099), and trade confirmations.
All data is fetched from database - NO MOCK DATA.
"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger
from cift.services.statement_generator import get_statement_service

router = APIRouter(prefix="/statements", tags=["statements"])


# ============================================================================
# MODELS
# ============================================================================

class AccountStatement(BaseModel):
    """Account statement model"""
    id: str
    user_id: str
    statement_type: str  # 'monthly', 'quarterly', 'annual'
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    file_url: str | None = None

    # Summary data
    starting_balance: Decimal
    ending_balance: Decimal
    total_deposits: Decimal
    total_withdrawals: Decimal
    total_trades: int
    realized_gain_loss: Decimal
    dividends_received: Decimal
    fees_paid: Decimal


class TaxDocument(BaseModel):
    """Tax document model"""
    id: str
    user_id: str
    document_type: str  # '1099-B', '1099-DIV', '1099-INT'
    tax_year: int
    generated_at: datetime
    file_url: str | None = None

    # Summary data
    total_proceeds: Decimal | None = None
    total_cost_basis: Decimal | None = None
    total_gain_loss: Decimal | None = None
    total_dividends: Decimal | None = None
    total_interest: Decimal | None = None


class TradeConfirmation(BaseModel):
    """Trade confirmation model"""
    id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: Decimal
    commission: Decimal
    executed_at: datetime
    settlement_date: datetime
    file_url: str | None = None


# ============================================================================
# ENDPOINTS - ACCOUNT STATEMENTS
# ============================================================================

@router.get("")
async def get_statements(
    year: int | None = None,
    statement_type: str | None = None,
    account_id: UUID | None = None,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get account statements using enhanced statement service"""

    try:
        statement_service = get_statement_service()
        statements = await statement_service.get_user_statements(
            user_id=user_id,
            account_id=account_id,
            statement_type=statement_type,
            limit=50
        )

        return {
            "statements": statements,
            "total": len(statements),
            "year": year,
            "statement_type": statement_type
        }

    except Exception as e:
        logger.error(f"Failed to get statements: {e}")

        # Fallback to database query"""
    pool = await get_postgres_pool()

    query = """
        SELECT
            id::text,
            user_id::text,
            statement_type,
            period_start,
            period_end,
            generated_at,
            file_url,
            starting_balance,
            ending_balance,
            total_deposits,
            total_withdrawals,
            total_trades,
            realized_gain_loss,
            dividends_received,
            fees_paid
        FROM account_statements
        WHERE user_id = $1
    """
    params = [user_id]
    param_count = 2

    if year:
        query += f" AND EXTRACT(YEAR FROM period_end) = ${param_count}"
        params.append(year)
        param_count += 1

    if statement_type:
        query += f" AND statement_type = ${param_count}"
        params.append(statement_type)
        param_count += 1

    query += " ORDER BY period_end DESC"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        return [
            AccountStatement(
                id=row['id'],
                user_id=row['user_id'],
                statement_type=row['statement_type'],
                period_start=row['period_start'],
                period_end=row['period_end'],
                generated_at=row['generated_at'],
                file_url=row['file_url'],
                starting_balance=row['starting_balance'],
                ending_balance=row['ending_balance'],
                total_deposits=row['total_deposits'],
                total_withdrawals=row['total_withdrawals'],
                total_trades=row['total_trades'],
                realized_gain_loss=row['realized_gain_loss'],
                dividends_received=row['dividends_received'],
                fees_paid=row['fees_paid'],
            )
            for row in rows
        ]


@router.post("/generate/monthly/{year}/{month}")
async def generate_monthly_statement(
    year: int,
    month: int,
    account_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """Generate monthly statement using enhanced service"""

    try:
        statement_service = get_statement_service()
        statement = await statement_service.generate_monthly_statement(
            user_id=user_id,
            account_id=account_id,
            year=year,
            month=month
        )

        return {
            "statement_id": str(statement.id),
            "generated_at": statement.generated_at,
            "file_size": statement.file_size,
            "page_count": statement.page_count,
            "message": "Monthly statement generated successfully"
        }

    except Exception as e:
        logger.error(f"Failed to generate monthly statement: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/generate/trade-confirmation/{order_id}")
async def generate_trade_confirmation(
    order_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """Generate trade confirmation document"""

    try:
        statement_service = get_statement_service()
        confirmation = await statement_service.generate_trade_confirmation(
            user_id=user_id,
            order_id=order_id
        )

        return {
            "confirmation_id": str(confirmation.id),
            "generated_at": confirmation.generated_at,
            "file_size": confirmation.file_size,
            "message": "Trade confirmation generated successfully"
        }

    except Exception as e:
        logger.error(f"Failed to generate trade confirmation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/generate/tax/{tax_year}")
async def generate_tax_document(
    tax_year: int,
    account_id: UUID,
    document_type: str = "1099",
    user_id: UUID = Depends(get_current_user_id),
):
    """Generate tax documents"""

    try:
        statement_service = get_statement_service()
        tax_doc = await statement_service.generate_tax_document(
            user_id=user_id,
            account_id=account_id,
            tax_year=tax_year,
            document_type=document_type
        )

        return {
            "document_id": str(tax_doc.id),
            "generated_at": tax_doc.generated_at,
            "file_size": tax_doc.file_size,
            "page_count": tax_doc.page_count,
            "document_type": document_type,
            "message": "Tax document generated successfully"
        }

    except Exception as e:
        logger.error(f"Failed to generate tax document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/generate/{statement_type}")
async def generate_statement_legacy(
    statement_type: str,
    period_start: datetime,
    period_end: datetime,
    user_id: UUID = Depends(get_current_user_id),
):
    """Generate account statement for a period (legacy endpoint)"""
    if statement_type not in ['monthly', 'quarterly', 'annual']:
        raise HTTPException(status_code=400, detail="Invalid statement type")

    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Get portfolio data for period
        portfolio = await conn.fetchrow(
            "SELECT cash, total_value FROM portfolios WHERE user_id = $1",
            user_id,
        )

        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        # Calculate statement metrics
        deposits = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM funding_transactions
            WHERE user_id = $1
            AND type = 'deposit'
            AND status = 'completed'
            AND completed_at BETWEEN $2 AND $3
            """,
            user_id,
            period_start,
            period_end,
        ) or Decimal("0")

        withdrawals = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM funding_transactions
            WHERE user_id = $1
            AND type = 'withdrawal'
            AND status = 'completed'
            AND completed_at BETWEEN $2 AND $3
            """,
            user_id,
            period_start,
            period_end,
        ) or Decimal("0")

        total_trades = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM orders
            WHERE user_id = $1
            AND status = 'filled'
            AND filled_at BETWEEN $2 AND $3
            """,
            user_id,
            period_start,
            period_end,
        ) or 0

        # Calculate realized gains/losses
        realized_gain_loss = await conn.fetchval(
            """
            SELECT COALESCE(SUM(realized_gain_loss), 0)
            FROM positions
            WHERE user_id = $1
            AND closed_at BETWEEN $2 AND $3
            """,
            user_id,
            period_start,
            period_end,
        ) or Decimal("0")

        # Calculate dividends
        dividends = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE user_id = $1
            AND type = 'dividend'
            AND created_at BETWEEN $2 AND $3
            """,
            user_id,
            period_start,
            period_end,
        ) or Decimal("0")

        # Calculate fees
        fees = await conn.fetchval(
            """
            SELECT COALESCE(SUM(commission), 0)
            FROM orders
            WHERE user_id = $1
            AND status = 'filled'
            AND filled_at BETWEEN $2 AND $3
            """,
            user_id,
            period_start,
            period_end,
        ) or Decimal("0")

        # Get starting balance (ending balance of previous period or initial)
        prev_statement = await conn.fetchrow(
            """
            SELECT ending_balance
            FROM account_statements
            WHERE user_id = $1 AND period_end < $2
            ORDER BY period_end DESC
            LIMIT 1
            """,
            user_id,
            period_start,
        )

        starting_balance = prev_statement['ending_balance'] if prev_statement else Decimal("0")
        ending_balance = portfolio['total_value']

        # Create statement record
        row = await conn.fetchrow(
            """
            INSERT INTO account_statements (
                user_id, statement_type, period_start, period_end,
                starting_balance, ending_balance, total_deposits, total_withdrawals,
                total_trades, realized_gain_loss, dividends_received, fees_paid
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id::text, generated_at
            """,
            user_id,
            statement_type,
            period_start,
            period_end,
            starting_balance,
            ending_balance,
            deposits,
            withdrawals,
            total_trades,
            realized_gain_loss,
            dividends,
            fees,
        )

        # TODO: Generate PDF document
        # file_url = await generate_statement_pdf(...)

        logger.info(f"Statement generated: id={row['id']}, user_id={user_id}, type={statement_type}")

        return {
            "statement_id": row['id'],
            "generated_at": row['generated_at'],
            "message": "Statement generated successfully",
        }


@router.get("/{statement_id}/download")
async def download_statement(
    statement_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get download URL for statement"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT file_url
            FROM account_statements
            WHERE id = $1::uuid AND user_id = $2
            """,
            statement_id,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="Statement not found")

        if not row['file_url']:
            raise HTTPException(status_code=404, detail="Statement file not available")

        return {"download_url": row['file_url']}


# ============================================================================
# ENDPOINTS - TAX DOCUMENTS
# ============================================================================

@router.get("/tax")
async def get_tax_documents(
    year: int | None = None,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get tax documents from database"""
    pool = await get_postgres_pool()

    query = """
        SELECT
            id::text,
            user_id::text,
            document_type,
            tax_year,
            generated_at,
            file_url,
            total_proceeds,
            total_cost_basis,
            total_gain_loss,
            total_dividends,
            total_interest
        FROM tax_documents
        WHERE user_id = $1
    """
    params = [user_id]

    if year:
        query += " AND tax_year = $2"
        params.append(year)

    query += " ORDER BY tax_year DESC, document_type"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        return [
            TaxDocument(
                id=row['id'],
                user_id=row['user_id'],
                document_type=row['document_type'],
                tax_year=row['tax_year'],
                generated_at=row['generated_at'],
                file_url=row['file_url'],
                total_proceeds=row['total_proceeds'],
                total_cost_basis=row['total_cost_basis'],
                total_gain_loss=row['total_gain_loss'],
                total_dividends=row['total_dividends'],
                total_interest=row['total_interest'],
            )
            for row in rows
        ]


@router.post("/tax/generate/{tax_year}")
async def generate_tax_forms(
    tax_year: int,
    user_id: UUID = Depends(get_current_user_id),
):
    """Generate tax forms (1099) for a year"""
    current_year = datetime.utcnow().year
    if tax_year > current_year or tax_year < current_year - 10:
        raise HTTPException(status_code=400, detail="Invalid tax year")

    pool = await get_postgres_pool()

    year_start = datetime(tax_year, 1, 1)
    year_end = datetime(tax_year, 12, 31, 23, 59, 59)

    async with pool.acquire() as conn:
        # Generate 1099-B (Capital gains/losses)
        proceeds = await conn.fetchval(
            """
            SELECT COALESCE(SUM(quantity * average_price), 0)
            FROM positions
            WHERE user_id = $1 AND closed_at BETWEEN $2 AND $3
            """,
            user_id,
            year_start,
            year_end,
        ) or Decimal("0")

        cost_basis = await conn.fetchval(
            """
            SELECT COALESCE(SUM(quantity * cost_basis), 0)
            FROM positions
            WHERE user_id = $1 AND closed_at BETWEEN $2 AND $3
            """,
            user_id,
            year_start,
            year_end,
        ) or Decimal("0")

        gain_loss = proceeds - cost_basis

        if proceeds > 0:
            await conn.execute(
                """
                INSERT INTO tax_documents (
                    user_id, document_type, tax_year,
                    total_proceeds, total_cost_basis, total_gain_loss
                ) VALUES ($1, '1099-B', $2, $3, $4, $5)
                ON CONFLICT (user_id, document_type, tax_year)
                DO UPDATE SET
                    total_proceeds = EXCLUDED.total_proceeds,
                    total_cost_basis = EXCLUDED.total_cost_basis,
                    total_gain_loss = EXCLUDED.total_gain_loss,
                    generated_at = NOW()
                """,
                user_id,
                tax_year,
                proceeds,
                cost_basis,
                gain_loss,
            )

        # Generate 1099-DIV (Dividends)
        dividends = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE user_id = $1
            AND type = 'dividend'
            AND created_at BETWEEN $2 AND $3
            """,
            user_id,
            year_start,
            year_end,
        ) or Decimal("0")

        if dividends > 0:
            await conn.execute(
                """
                INSERT INTO tax_documents (
                    user_id, document_type, tax_year, total_dividends
                ) VALUES ($1, '1099-DIV', $2, $3)
                ON CONFLICT (user_id, document_type, tax_year)
                DO UPDATE SET
                    total_dividends = EXCLUDED.total_dividends,
                    generated_at = NOW()
                """,
                user_id,
                tax_year,
                dividends,
            )

        # Generate 1099-INT (Interest - if applicable)
        interest = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE user_id = $1
            AND type = 'interest'
            AND created_at BETWEEN $2 AND $3
            """,
            user_id,
            year_start,
            year_end,
        ) or Decimal("0")

        if interest > 0:
            await conn.execute(
                """
                INSERT INTO tax_documents (
                    user_id, document_type, tax_year, total_interest
                ) VALUES ($1, '1099-INT', $2, $3)
                ON CONFLICT (user_id, document_type, tax_year)
                DO UPDATE SET
                    total_interest = EXCLUDED.total_interest,
                    generated_at = NOW()
                """,
                user_id,
                tax_year,
                interest,
            )

        logger.info(f"Tax forms generated: user_id={user_id}, year={tax_year}")

        # TODO: Generate PDF documents

        return {
            "tax_year": tax_year,
            "message": "Tax forms generated successfully",
            "forms_generated": {
                "1099-B": proceeds > 0,
                "1099-DIV": dividends > 0,
                "1099-INT": interest > 0,
            },
        }


@router.get("/tax/{document_id}/download")
async def download_tax_document(
    document_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get download URL for tax document"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT file_url
            FROM tax_documents
            WHERE id = $1::uuid AND user_id = $2
            """,
            document_id,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="Tax document not found")

        if not row['file_url']:
            raise HTTPException(status_code=404, detail="Document file not available")

        return {"download_url": row['file_url']}
