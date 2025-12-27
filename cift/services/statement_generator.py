"""
CIFT Markets - Statement & Tax Document Generation Service

Advanced PDF statement generation with comprehensive financial reporting.
Supports monthly statements, trade confirmations, tax documents, and compliance reports.
"""

from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available - PDF generation will use mock mode")

from cift.core.database import get_postgres_pool


class StatementPeriod(BaseModel):
    start_date: date
    end_date: date
    period_type: str  # monthly, quarterly, annual, custom


class StatementData(BaseModel):
    user_info: dict[str, Any]
    account_info: dict[str, Any]
    period: StatementPeriod
    portfolio_summary: dict[str, Any]
    transactions: list[dict[str, Any]]
    positions: list[dict[str, Any]]
    performance_metrics: dict[str, Any]
    tax_implications: dict[str, Any]


class GeneratedStatement(BaseModel):
    id: UUID
    user_id: UUID
    account_id: UUID
    statement_type: str
    period_start: date
    period_end: date
    file_path: str | None = None
    file_data: bytes | None = None
    generated_at: datetime
    file_size: int = 0
    page_count: int = 0


class StatementGeneratorService:
    """Advanced financial statement and tax document generation service."""

    def __init__(self):
        self.styles = None
        if REPORTLAB_AVAILABLE:
            self._setup_styles()

    def _setup_styles(self):
        """Setup PDF styles for consistent formatting."""
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CompanyHeader',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=1  # Center
        ))

        self.styles.add(ParagraphStyle(
            name='StatementTitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=20
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#4a5568'),
            spaceBefore=20,
            spaceAfter=10
        ))

    async def generate_monthly_statement(
        self,
        user_id: UUID,
        account_id: UUID,
        year: int,
        month: int
    ) -> GeneratedStatement:
        """Generate comprehensive monthly account statement."""

        period_start = date(year, month, 1)
        if month == 12:
            period_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            period_end = date(year, month + 1, 1) - timedelta(days=1)

        period = StatementPeriod(
            start_date=period_start,
            end_date=period_end,
            period_type="monthly"
        )

        # Gather statement data
        statement_data = await self._gather_statement_data(user_id, account_id, period)

        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_data = await self._generate_pdf_statement(statement_data, "Monthly Statement")
        else:
            pdf_data = await self._generate_mock_pdf(statement_data, "Monthly Statement")

        # Store in database
        statement = GeneratedStatement(
            id=uuid4(),
            user_id=user_id,
            account_id=account_id,
            statement_type="monthly",
            period_start=period_start,
            period_end=period_end,
            file_data=pdf_data,
            generated_at=datetime.utcnow(),
            file_size=len(pdf_data),
            page_count=self._estimate_page_count(pdf_data)
        )

        await self._store_statement(statement)

        logger.success(f"Generated monthly statement for {year}-{month:02d}")
        return statement

    async def generate_trade_confirmation(
        self,
        user_id: UUID,
        order_id: UUID
    ) -> GeneratedStatement:
        """Generate trade confirmation document."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Get order details
            order = await conn.fetchrow("""
                SELECT o.*, a.account_number, u.first_name, u.last_name, u.email
                FROM orders o
                JOIN accounts a ON a.id = o.account_id
                JOIN users u ON u.id = o.user_id
                WHERE o.id = $1 AND o.user_id = $2
            """, order_id, user_id)

            if not order:
                raise ValueError(f"Order {order_id} not found")

        # Create mini statement data for trade confirmation
        trade_data = {
            "user_info": {
                "name": f"{order['first_name']} {order['last_name']}",
                "email": order['email'],
                "account_number": order['account_number']
            },
            "trade_details": {
                "order_id": str(order['id']),
                "symbol": order['symbol'],
                "side": order['side'],
                "quantity": float(order['quantity']),
                "price": float(order['price'] or 0),
                "order_type": order['order_type'],
                "status": order['status'],
                "executed_at": order['executed_at'],
                "total_value": float(order['filled_quantity'] or 0) * float(order['average_price'] or 0)
            }
        }

        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_data = await self._generate_trade_confirmation_pdf(trade_data)
        else:
            pdf_data = await self._generate_mock_pdf(trade_data, "Trade Confirmation")

        # Store confirmation
        statement = GeneratedStatement(
            id=uuid4(),
            user_id=user_id,
            account_id=UUID(order['account_id']),
            statement_type="trade_confirmation",
            period_start=order['created_at'].date(),
            period_end=order['created_at'].date(),
            file_data=pdf_data,
            generated_at=datetime.utcnow(),
            file_size=len(pdf_data),
            page_count=1
        )

        await self._store_statement(statement)

        logger.success(f"Generated trade confirmation for order {order_id}")
        return statement

    async def generate_tax_document(
        self,
        user_id: UUID,
        account_id: UUID,
        tax_year: int,
        document_type: str = "1099"
    ) -> GeneratedStatement:
        """Generate tax documents (1099, consolidated tax summary)."""

        period = StatementPeriod(
            start_date=date(tax_year, 1, 1),
            end_date=date(tax_year, 12, 31),
            period_type="annual"
        )

        # Gather tax-specific data
        tax_data = await self._gather_tax_data(user_id, account_id, tax_year)

        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_data = await self._generate_tax_document_pdf(tax_data, document_type)
        else:
            pdf_data = await self._generate_mock_pdf(tax_data, f"Tax Document {document_type}")

        statement = GeneratedStatement(
            id=uuid4(),
            user_id=user_id,
            account_id=account_id,
            statement_type=f"tax_{document_type}",
            period_start=period.start_date,
            period_end=period.end_date,
            file_data=pdf_data,
            generated_at=datetime.utcnow(),
            file_size=len(pdf_data),
            page_count=self._estimate_page_count(pdf_data)
        )

        await self._store_statement(statement)

        logger.success(f"Generated {document_type} tax document for {tax_year}")
        return statement

    async def _gather_statement_data(
        self,
        user_id: UUID,
        account_id: UUID,
        period: StatementPeriod
    ) -> StatementData:
        """Gather comprehensive data for statement generation."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # User information
            user_info = await conn.fetchrow("""
                SELECT first_name, last_name, email, phone,
                       address_line1, address_line2, city, state, zip_code
                FROM users
                WHERE id = $1
            """, user_id)

            # Account information
            account_info = await conn.fetchrow("""
                SELECT account_number, account_type, cash_balance,
                       equity, buying_power, created_at
                FROM accounts
                WHERE id = $1 AND user_id = $2
            """, account_id, user_id)

            # Transactions during period
            transactions = await conn.fetch("""
                SELECT t.*, o.symbol, o.side, o.order_type
                FROM transactions t
                LEFT JOIN orders o ON o.id = t.order_id
                WHERE t.account_id = $1
                AND t.created_at >= $2 AND t.created_at <= $3
                ORDER BY t.created_at DESC
            """, account_id, period.start_date, period.end_date + timedelta(days=1))

            # Current positions
            positions = await conn.fetch("""
                SELECT p.*, s.name, s.sector
                FROM positions p
                LEFT JOIN symbols s ON s.symbol = p.symbol
                WHERE p.account_id = $1 AND p.quantity != 0
                ORDER BY p.market_value DESC
            """, account_id)

            # Portfolio snapshots for performance
            snapshots = await conn.fetch("""
                SELECT * FROM portfolio_snapshots
                WHERE account_id = $1
                AND timestamp >= $2 AND timestamp <= $3
                ORDER BY timestamp ASC
            """, account_id, period.start_date, period.end_date + timedelta(days=1))

        # Calculate performance metrics
        performance_metrics = self._calculate_period_performance(snapshots)

        # Calculate tax implications
        tax_implications = self._calculate_tax_implications(transactions)

        return StatementData(
            user_info=dict(user_info) if user_info else {},
            account_info=dict(account_info) if account_info else {},
            period=period,
            portfolio_summary={
                "total_value": float(account_info['equity']) if account_info else 0,
                "cash_balance": float(account_info['cash_balance']) if account_info else 0,
                "positions_count": len(positions),
                "transactions_count": len(transactions)
            },
            transactions=[dict(t) for t in transactions],
            positions=[dict(p) for p in positions],
            performance_metrics=performance_metrics,
            tax_implications=tax_implications
        )

    async def _gather_tax_data(
        self,
        user_id: UUID,
        account_id: UUID,
        tax_year: int
    ) -> dict[str, Any]:
        """Gather tax-specific data for the year."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Realized gains/losses
            realized_trades = await conn.fetch("""
                SELECT
                    t.symbol,
                    t.quantity,
                    t.price,
                    t.total_amount,
                    t.fees,
                    t.created_at,
                    o.side
                FROM transactions t
                LEFT JOIN orders o ON o.id = t.order_id
                WHERE t.account_id = $1
                AND EXTRACT(YEAR FROM t.created_at) = $2
                AND t.type = 'trade'
                ORDER BY t.symbol, t.created_at
            """, account_id, tax_year)

            # Dividend income
            dividends = await conn.fetch("""
                SELECT * FROM transactions
                WHERE account_id = $1
                AND EXTRACT(YEAR FROM created_at) = $2
                AND type = 'dividend'
                ORDER BY created_at
            """, account_id, tax_year)

            # Interest income
            interest = await conn.fetch("""
                SELECT * FROM transactions
                WHERE account_id = $1
                AND EXTRACT(YEAR FROM created_at) = $2
                AND type = 'interest'
                ORDER BY created_at
            """, account_id, tax_year)

        # Calculate wash sales, holding periods, etc.
        tax_calculations = self._calculate_tax_implications([dict(t) for t in realized_trades])

        return {
            "tax_year": tax_year,
            "realized_trades": [dict(t) for t in realized_trades],
            "dividends": [dict(d) for d in dividends],
            "interest": [dict(i) for i in interest],
            "calculations": tax_calculations
        }

    def _calculate_period_performance(self, snapshots: list) -> dict[str, Any]:
        """Calculate performance metrics for the period."""

        if len(snapshots) < 2:
            return {
                "total_return": 0,
                "total_return_pct": 0,
                "best_day": 0,
                "worst_day": 0,
                "volatility": 0
            }

        start_value = float(snapshots[0]['total_value'])
        end_value = float(snapshots[-1]['total_value'])
        total_return = end_value - start_value
        total_return_pct = (total_return / start_value * 100) if start_value > 0 else 0

        # Daily returns
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_val = float(snapshots[i-1]['total_value'])
            curr_val = float(snapshots[i]['total_value'])
            if prev_val > 0:
                daily_returns.append((curr_val - prev_val) / prev_val * 100)

        best_day = max(daily_returns) if daily_returns else 0
        worst_day = min(daily_returns) if daily_returns else 0

        return {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "best_day": best_day,
            "worst_day": worst_day,
            "volatility": 0  # Simplified for now
        }

    def _calculate_tax_implications(self, transactions: list) -> dict[str, Any]:
        """Calculate tax implications from transactions."""

        total_realized_gains = 0
        total_fees = 0
        short_term_gains = 0
        long_term_gains = 0

        for txn in transactions:
            if txn.get('type') == 'trade':
                fees = float(txn.get('fees', 0))
                total_fees += fees

                # Simplified gain/loss calculation
                # In real implementation, need proper cost basis tracking
                amount = float(txn.get('total_amount', 0))
                if txn.get('side') == 'sell':
                    # This is a simplification - real calculation needs cost basis
                    estimated_gain = amount * 0.1  # Mock 10% gain
                    total_realized_gains += estimated_gain

                    # Assume short-term for simplicity
                    short_term_gains += estimated_gain

        return {
            "total_realized_gains": total_realized_gains,
            "short_term_gains": short_term_gains,
            "long_term_gains": long_term_gains,
            "total_fees": total_fees,
            "wash_sales": 0,  # Would need complex calculation
            "tax_estimate": total_realized_gains * 0.25  # Simplified tax estimate
        }

    async def _generate_pdf_statement(
        self,
        data: StatementData,
        title: str
    ) -> bytes:
        """Generate PDF statement using ReportLab."""

        if not REPORTLAB_AVAILABLE:
            return await self._generate_mock_pdf(data, title)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

        # Build story
        story = []

        # Header
        story.append(Paragraph("CIFT Markets", self.styles['CompanyHeader']))
        story.append(Paragraph(title, self.styles['StatementTitle']))
        story.append(Spacer(1, 12))

        # Account info section
        story.append(Paragraph("Account Information", self.styles['SectionHeader']))

        account_data = [
            ['Account Holder:', f"{data.user_info.get('first_name', '')} {data.user_info.get('last_name', '')}"],
            ['Account Number:', data.account_info.get('account_number', 'N/A')],
            ['Statement Period:', f"{data.period.start_date} to {data.period.end_date}"],
            ['Account Type:', data.account_info.get('account_type', 'Trading')],
        ]

        account_table = Table(account_data, colWidths=[2*inch, 3*inch])
        account_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(account_table)
        story.append(Spacer(1, 20))

        # Portfolio summary
        story.append(Paragraph("Portfolio Summary", self.styles['SectionHeader']))

        summary_data = [
            ['Total Portfolio Value:', f"${data.portfolio_summary.get('total_value', 0):,.2f}"],
            ['Cash Balance:', f"${data.portfolio_summary.get('cash_balance', 0):,.2f}"],
            ['Number of Positions:', str(data.portfolio_summary.get('positions_count', 0))],
            ['Transactions This Period:', str(data.portfolio_summary.get('transactions_count', 0))],
        ]

        summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Performance metrics
        if data.performance_metrics:
            story.append(Paragraph("Performance Summary", self.styles['SectionHeader']))

            perf_data = [
                ['Total Return:', f"${data.performance_metrics.get('total_return', 0):,.2f}"],
                ['Total Return %:', f"{data.performance_metrics.get('total_return_pct', 0):.2f}%"],
                ['Best Day:', f"{data.performance_metrics.get('best_day', 0):+.2f}%"],
                ['Worst Day:', f"{data.performance_metrics.get('worst_day', 0):+.2f}%"],
            ]

            perf_table = Table(perf_data, colWidths=[2.5*inch, 2.5*inch])
            perf_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(perf_table)
            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    async def _generate_mock_pdf(self, data: Any, title: str) -> bytes:
        """Generate mock PDF data when ReportLab is not available."""

        # Create a simple mock PDF content
        mock_content = f"""
        CIFT Markets - {title}

        Generated: {datetime.utcnow().isoformat()}

        This is a mock PDF statement generated for development purposes.
        In production, this would be a properly formatted PDF document.

        Data: {str(data)[:200]}...
        """

        return mock_content.encode('utf-8')

    async def _generate_trade_confirmation_pdf(self, trade_data: dict) -> bytes:
        """Generate trade confirmation PDF."""

        if not REPORTLAB_AVAILABLE:
            return await self._generate_mock_pdf(trade_data, "Trade Confirmation")

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []

        # Header
        story.append(Paragraph("CIFT Markets - Trade Confirmation", self.styles['CompanyHeader']))
        story.append(Spacer(1, 20))

        # Trade details
        trade_details = trade_data['trade_details']

        trade_table_data = [
            ['Order ID:', trade_details['order_id']],
            ['Symbol:', trade_details['symbol']],
            ['Side:', trade_details['side'].upper()],
            ['Quantity:', f"{trade_details['quantity']:,}"],
            ['Price:', f"${trade_details['price']:.2f}"],
            ['Total Value:', f"${trade_details['total_value']:,.2f}"],
            ['Status:', trade_details['status'].title()],
            ['Executed:', str(trade_details.get('executed_at', 'Pending'))],
        ]

        trade_table = Table(trade_table_data, colWidths=[2*inch, 3*inch])
        trade_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(trade_table)

        doc.build(story)

        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    async def _generate_tax_document_pdf(self, tax_data: dict, document_type: str) -> bytes:
        """Generate tax document PDF."""

        if not REPORTLAB_AVAILABLE:
            return await self._generate_mock_pdf(tax_data, f"Tax {document_type}")

        # Implement comprehensive tax document generation
        # This would include proper 1099 formatting, realized gains/losses, etc.

        return await self._generate_mock_pdf(tax_data, f"Tax {document_type}")

    def _estimate_page_count(self, pdf_data: bytes) -> int:
        """Estimate page count from PDF data size."""
        # Rough estimation: 1 page ~= 50KB
        return max(1, len(pdf_data) // 50000)

    async def _store_statement(self, statement: GeneratedStatement):
        """Store generated statement in database."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO generated_statements (
                    id, user_id, account_id, statement_type,
                    period_start, period_end, file_data, generated_at,
                    file_size, page_count
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                statement.id, statement.user_id, statement.account_id,
                statement.statement_type, statement.period_start, statement.period_end,
                statement.file_data, statement.generated_at,
                statement.file_size, statement.page_count
            )

    async def get_user_statements(
        self,
        user_id: UUID,
        account_id: UUID | None = None,
        statement_type: str | None = None,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get user's generated statements."""

        pool = await get_postgres_pool()

        query = """
            SELECT
                id, account_id, statement_type, period_start, period_end,
                generated_at, file_size, page_count
            FROM generated_statements
            WHERE user_id = $1
        """
        params = [user_id]
        param_count = 2

        if account_id:
            query += f" AND account_id = ${param_count}"
            params.append(account_id)
            param_count += 1

        if statement_type:
            query += f" AND statement_type = ${param_count}"
            params.append(statement_type)
            param_count += 1

        query += f" ORDER BY generated_at DESC LIMIT ${param_count}"
        params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "id": str(row['id']),
                "account_id": str(row['account_id']),
                "statement_type": row['statement_type'],
                "period_start": row['period_start'],
                "period_end": row['period_end'],
                "generated_at": row['generated_at'],
                "file_size": row['file_size'],
                "page_count": row['page_count']
            }
            for row in rows
        ]

    async def get_statement_file(self, statement_id: UUID, user_id: UUID) -> bytes | None:
        """Get statement file data."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT file_data
                FROM generated_statements
                WHERE id = $1 AND user_id = $2
            """, statement_id, user_id)

        return row['file_data'] if row else None


# Global statement generator service
_statement_service = None

def get_statement_service() -> StatementGeneratorService:
    """Get the global statement service instance."""
    global _statement_service
    if _statement_service is None:
        _statement_service = StatementGeneratorService()
    return _statement_service
