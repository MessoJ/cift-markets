"""
Receipt Generator Service - RULES COMPLIANT
Generates premium PDF receipts for funding transactions
Professional financial document design inspired by top trading platforms
"""
import logging
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)

# PDF generation library - install with: pip install reportlab qrcode pillow
try:
    from reportlab.graphics import renderPDF
    from reportlab.graphics.shapes import Circle, Drawing, Line, Rect
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        Frame,
        Image,
        KeepTogether,
        PageBreak,
        PageTemplate,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
    logger.info("ReportLab loaded successfully")
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    logger.error(f"ReportLab import error: {e}")

try:
    import qrcode
    QRCODE_AVAILABLE = True
    logger.info("QRCode loaded successfully")
except ImportError as e:
    QRCODE_AVAILABLE = False
    logger.warning(f"QRCode not available: {e}")


class ReceiptGenerator:
    """Generate premium PDF receipts for transactions - RULES COMPLIANT"""

    # Brand colors matching the platform
    PRIMARY_BLUE = colors.HexColor('#3b82f6')
    DARK_BLUE = colors.HexColor('#2563eb')
    SUCCESS_GREEN = colors.HexColor('#22c55e')
    DANGER_RED = colors.HexColor('#ef4444')
    BACKGROUND_GRAY = colors.HexColor('#f8fafc')
    BORDER_GRAY = colors.HexColor('#e2e8f0')
    TEXT_DARK = colors.HexColor('#1e293b')
    TEXT_GRAY = colors.HexColor('#64748b')

    @staticmethod
    def _create_qr_code(data: str) -> BytesIO:
        """Generate QR code for transaction verification"""
        if not QRCODE_AVAILABLE:
            raise ImportError("QRCode library not available")

        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=2)
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            logger.info(f"QR code generated successfully for: {data[:50]}")
            return buffer
        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            raise

    @staticmethod
    async def generate_receipt(
        transaction_data: dict[str, Any],
        user_data: dict[str, Any],
        payment_method_data: dict[str, Any]
    ) -> BytesIO:
        """
        Generate premium PDF receipt from database transaction data
        Professional design inspired by top financial platforms

        Args:
            transaction_data: Transaction from funding_transactions table
            user_data: User from users table
            payment_method_data: Payment method from payment_methods table

        Returns:
            BytesIO: PDF file as bytes
        """
        if not REPORTLAB_AVAILABLE:
            error_msg = "reportlab not installed. Install with: pip install reportlab qrcode pillow"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            logger.info(f"Generating receipt for transaction: {transaction_data.get('id', 'unknown')}")

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.5*inch,
                bottomMargin=0.75*inch
            )
            story = []
            styles = getSampleStyleSheet()
        except Exception as e:
            logger.error(f"Error initializing PDF document: {e}")
            raise

        # Extract transaction data
        transaction_id = str(transaction_data.get('id', 'N/A'))
        transaction_type = transaction_data.get('type', '').upper()
        amount = transaction_data.get('amount', Decimal('0.00'))
        fee = transaction_data.get('fee', Decimal('0.00'))
        status = transaction_data.get('status', '').capitalize()
        created_at = transaction_data.get('created_at')

        # Format date
        if isinstance(created_at, datetime):
            date_str = created_at.strftime('%B %d, %Y')
            time_str = created_at.strftime('%I:%M %p %Z')
        else:
            date_str = str(created_at)
            time_str = ''

        # Professional brokerage-style typography system
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Normal'],
            fontSize=20,
            textColor=ReceiptGenerator.TEXT_DARK,
            fontName='Helvetica-Bold',
            spaceAfter=8,
            alignment=TA_CENTER
        )

        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=ReceiptGenerator.TEXT_GRAY,
            fontName='Helvetica',
            spaceAfter=6,
            alignment=TA_CENTER
        )

        section_title_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=ReceiptGenerator.PRIMARY_BLUE,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=10,
            borderPadding=(4, 0, 4, 0)
        )

        label_style = ParagraphStyle(
            'Label',
            parent=styles['Normal'],
            fontSize=8,
            textColor=ReceiptGenerator.TEXT_GRAY,
            fontName='Helvetica',
            leading=11
        )

        value_style = ParagraphStyle(
            'Value',
            parent=styles['Normal'],
            fontSize=9,
            textColor=ReceiptGenerator.TEXT_DARK,
            fontName='Helvetica-Bold',
            leading=11
        )

        amount_label_style = ParagraphStyle(
            'AmountLabel',
            parent=styles['Normal'],
            fontSize=9,
            textColor=ReceiptGenerator.TEXT_GRAY,
            fontName='Helvetica',
            leading=12
        )

        amount_value_style = ParagraphStyle(
            'AmountValue',
            parent=styles['Normal'],
            fontSize=9,
            textColor=ReceiptGenerator.TEXT_DARK,
            fontName='Helvetica',
            leading=12
        )

        # === PROFESSIONAL HEADER WITH METADATA ===
        # Logo - matching Logo component (20px for prominence)
        logo_html = '<para align="center"><font name="Helvetica-Bold" size="20" color="#3b82f6">CIFT</font> <font name="Helvetica" size="20" color="#1e293b">MARKETS</font></para>'
        story.append(Paragraph(logo_html, header_style))
        story.append(Spacer(1, 0.02*inch))  # Extra space to prevent overlap
        story.append(Paragraph('Advanced Trading Platform', subtitle_style))

        # Decorative line
        line_drawing = Drawing(6.5*inch, 2)
        line_drawing.add(Line(0, 1, 6.5*inch, 1, strokeColor=ReceiptGenerator.PRIMARY_BLUE, strokeWidth=1.5))
        story.append(line_drawing)
        story.append(Spacer(1, 0.08*inch))

        # Receipt title and document info in one row
        doc_id_style = ParagraphStyle('DocID', parent=subtitle_style, fontSize=7, alignment=TA_RIGHT)

        header_row = [
            [
                Paragraph('<b>TRANSACTION RECEIPT</b>', section_title_style),
                Paragraph(f'Document ID: {transaction_id[:16]}', doc_id_style)
            ]
        ]
        header_table = Table(header_row, colWidths=[3.8*inch, 2.7*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.12*inch))

        # Calculate values
        total_amount = amount + fee if transaction_type == 'DEPOSIT' else amount
        transaction_type_display = 'Deposit' if transaction_type == 'DEPOSIT' else 'Withdrawal'

        # === MULTI-COLUMN LAYOUT - Professional brokerage style ===
        pm_type = payment_method_data.get('type', 'Unknown').replace('_', ' ').title()
        pm_last4 = payment_method_data.get('last_four', '****')
        pm_name = payment_method_data.get('name', 'N/A')
        pm_display = f"{pm_type} ••••{pm_last4}"
        if pm_name and pm_name != 'N/A':
            pm_display += f" ({pm_name})"

        user_name = user_data.get('full_name', 'N/A')
        user_email = user_data.get('email', 'N/A')

        # Status indicator with color
        status_display = '✓ ' + status.upper() if status.lower() == 'completed' else status.upper()
        status_color = ReceiptGenerator.SUCCESS_GREEN if status.lower() == 'completed' else ReceiptGenerator.TEXT_GRAY

        # === LEFT COLUMN: Transaction Details ===
        left_col_data = []

        # Transaction Information Section
        left_col_data.append([Paragraph('<b>TRANSACTION DETAILS</b>', section_title_style)])
        left_col_data.append([Paragraph('Transaction ID', label_style)])
        left_col_data.append([Paragraph(transaction_id[:24] + '...', value_style)])
        left_col_data.append([Paragraph('Date & Time', label_style)])
        left_col_data.append([Paragraph(f'{date_str}<br/>{time_str}', value_style)])
        left_col_data.append([Paragraph('Type', label_style)])
        left_col_data.append([Paragraph(f'<font color="{status_color.hexval()}"><b>{transaction_type_display.upper()}</b></font>', value_style)])
        left_col_data.append([Paragraph('', label_style)])  # Spacer

        # Payment Method Section
        left_col_data.append([Paragraph('<b>PAYMENT METHOD</b>', section_title_style)])
        left_col_data.append([Paragraph('Method', label_style)])
        left_col_data.append([Paragraph(pm_display, value_style)])
        left_col_data.append([Paragraph('', label_style)])  # Spacer

        # Account Information Section
        left_col_data.append([Paragraph('<b>ACCOUNT INFORMATION</b>', section_title_style)])
        left_col_data.append([Paragraph('Account Holder', label_style)])
        left_col_data.append([Paragraph(user_name, value_style)])
        left_col_data.append([Paragraph('Email Address', label_style)])
        left_col_data.append([Paragraph(user_email, value_style)])

        left_table = Table(left_col_data, colWidths=[3*inch])
        left_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        # === RIGHT COLUMN: Status & Amount Summary ===
        right_col_data = []

        # Status Badge
        status_badge = Paragraph(
            f'<para align="center" backColor="{status_color.hexval()}">'
            f'<font color="white" size="9"><b>  {status_display}  </b></font></para>',
            value_style
        )
        right_col_data.append([status_badge])
        right_col_data.append([Paragraph('', label_style)])  # Spacer

        # Amount Display (Prominent)
        amount_display = f'{"+" if transaction_type == "DEPOSIT" else "-"}${total_amount:,.2f}'
        amount_color = ReceiptGenerator.SUCCESS_GREEN if transaction_type == 'DEPOSIT' else ReceiptGenerator.DANGER_RED
        amount_para = Paragraph(
            f'<para align="center"><font color="{amount_color.hexval()}" size="20"><b>{amount_display}</b></font></para>',
            value_style
        )
        right_col_data.append([amount_para])
        right_col_data.append([Paragraph(f'<para align="center"><font size="8" color="#64748b">{transaction_type_display}</font></para>', value_style)])
        right_col_data.append([Paragraph('', label_style)])  # Spacer
        right_col_data.append([Paragraph('', label_style)])  # Spacer

        # Amount Breakdown Box
        right_col_data.append([Paragraph('<b>AMOUNT BREAKDOWN</b>', section_title_style)])
        breakdown_data = [
            [Paragraph('Subtotal', amount_label_style), Paragraph(f'${amount:,.2f}', amount_value_style)],
            [Paragraph('Processing Fee', amount_label_style), Paragraph(f'${fee:,.2f}', amount_value_style)],
            [Paragraph('<b>Total</b>', ParagraphStyle('BoldLabel', parent=amount_label_style, fontName='Helvetica-Bold')),
             Paragraph(f'<font color="{ReceiptGenerator.PRIMARY_BLUE.hexval()}"><b>${total_amount:,.2f}</b></font>', amount_value_style)],
        ]
        breakdown_table = Table(breakdown_data, colWidths=[1.5*inch, 1.5*inch])
        breakdown_table.setStyle(TableStyle([
            ('LINEABOVE', (0, 2), (-1, 2), 1.5, ReceiptGenerator.PRIMARY_BLUE),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        right_col_data.append([breakdown_table])

        right_table = Table(right_col_data, colWidths=[3.2*inch])
        right_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReceiptGenerator.BACKGROUND_GRAY),
            ('BOX', (0, 0), (-1, -1), 1, ReceiptGenerator.BORDER_GRAY),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        # === COMBINE INTO 2-COLUMN LAYOUT ===
        main_layout = [[left_table, right_table]]
        main_table = Table(main_layout, colWidths=[3.1*inch, 3.4*inch], spaceBefore=0, spaceAfter=0)
        main_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))

        story.append(main_table)
        story.append(Spacer(1, 0.12*inch))

        # === QR CODE & VERIFICATION SECTION ===
        qr_section_data = []

        # Generate QR code linking to verification page
        qr_cell = None
        try:
            if QRCODE_AVAILABLE:
                # QR code links to public verification page
                qr_data = f"http://localhost:3000/verify/{transaction_id}"
                qr_buffer = ReceiptGenerator._create_qr_code(qr_data)
                qr_image = Image(qr_buffer, width=0.8*inch, height=0.8*inch)
                qr_cell = qr_image
            else:
                raise ImportError("QRCode not available")
        except Exception as e:
            logger.warning(f"QR code generation skipped: {e}")
            # Fallback to text box
            qr_cell = Paragraph(
                '<para align="center"><font size="7" color="#64748b">Verification<br/>QR Code</font></para>',
                subtitle_style
            )

        # Verification info
        verification_info = Paragraph(
            f'<para align="center"><font size="7" color="#64748b">'
            f'<b>Transaction Verification</b><br/>'
            f'Scan QR code to verify this transaction<br/>'
            f'or visit: <b>ciftmarkets.com/verify</b><br/>'
            f'Transaction ID: <b>{transaction_id[:24]}</b>'
            f'</font></para>',
            subtitle_style
        )

        qr_section_data = [[qr_cell, verification_info]]
        qr_table = Table(qr_section_data, colWidths=[1*inch, 5.5*inch])
        qr_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(qr_table)
        story.append(Spacer(1, 0.1*inch))

        # === COMPREHENSIVE FOOTER ===
        footer_line = Drawing(6.5*inch, 1)
        footer_line.add(Line(0, 0, 6.5*inch, 0, strokeColor=ReceiptGenerator.BORDER_GRAY, strokeWidth=1))
        story.append(footer_line)
        story.append(Spacer(1, 0.08*inch))

        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=7,
            textColor=ReceiptGenerator.TEXT_GRAY,
            alignment=TA_CENTER,
            leading=9
        )

        footer_text = f"""
        <para align="center">
        <b>CIFT Markets</b> • Advanced Trading Platform<br/>
        support@ciftmarkets.com • www.ciftmarkets.com • +1 (646) 978-2187<br/>
        <br/>
        This document serves as an official receipt for your transaction.<br/>
        Please retain this receipt for your financial records and tax purposes.<br/>
        <br/>
        Securities and derivatives trading involves risk of loss. Past performance does not guarantee future results.<br/>
        CIFT Markets is a member of FINRA and SIPC. © {datetime.now().year} CIFT Markets. All rights reserved.<br/>
        <br/>
        Document Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} UTC | Document ID: {transaction_id[:20]}
        </para>
        """

        story.append(Paragraph(footer_text, footer_style))

        # Build PDF
        try:
            logger.info("Building PDF document...")
            doc.build(story)
            buffer.seek(0)
            logger.info(f"Receipt generated successfully. Size: {buffer.getbuffer().nbytes} bytes")
            return buffer
        except Exception as e:
            logger.error(f"Error building PDF document: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate PDF receipt: {str(e)}") from e

    @staticmethod
    def generate_simple_text_receipt(
        transaction_data: dict[str, Any],
        user_data: dict[str, Any],
        payment_method_data: dict[str, Any]
    ) -> str:
        """
        Generate plain text receipt as fallback if reportlab not available

        Returns:
            str: Plain text receipt
        """
        lines = []
        lines.append("=" * 60)
        lines.append("              CIFT MARKETS - RECEIPT")
        lines.append("=" * 60)
        lines.append("")

        transaction_id = transaction_data.get('id', 'N/A')
        transaction_type = transaction_data.get('type', '').upper()
        amount = transaction_data.get('amount', Decimal('0.00'))
        fee = transaction_data.get('fee', Decimal('0.00'))
        total = amount + fee if transaction_type == 'DEPOSIT' else amount - fee
        status = transaction_data.get('status', '').upper()
        created_at = transaction_data.get('created_at')

        if isinstance(created_at, datetime):
            date_str = created_at.strftime('%B %d, %Y at %I:%M %p')
        else:
            date_str = str(created_at)

        lines.append(f"Transaction ID: {transaction_id}")
        lines.append(f"Type: {transaction_type}")
        lines.append(f"Status: {status}")
        lines.append(f"Date: {date_str}")
        lines.append("")
        lines.append(f"Amount: ${amount:.2f}")
        lines.append(f"Fee: ${fee:.2f}")
        lines.append("-" * 60)
        lines.append(f"Total: ${total:.2f}")
        lines.append("")

        pm_type = payment_method_data.get('type', 'Unknown')
        pm_name = payment_method_data.get('name', 'N/A')
        pm_last4 = payment_method_data.get('last_four', '****')

        lines.append("PAYMENT METHOD")
        lines.append(f"Type: {pm_type.replace('_', ' ').title()}")
        lines.append(f"Name: {pm_name}")
        lines.append(f"Ending in: {pm_last4}")
        lines.append("")

        user_name = user_data.get('full_name', 'N/A')
        user_email = user_data.get('email', 'N/A')

        lines.append("ACCOUNT HOLDER")
        lines.append(f"Name: {user_name}")
        lines.append(f"Email: {user_email}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("For questions, contact support@ciftmarkets.com")
        lines.append(f"Generated on {datetime.now().strftime('%B %d, %Y')}")
        lines.append("=" * 60)

        return "\n".join(lines)
