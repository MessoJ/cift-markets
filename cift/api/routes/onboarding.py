"""
KYC/ONBOARDING API ROUTES
Handles user verification, identity documents, and compliance.
All data is fetched from database - NO MOCK DATA.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from cift.core.auth import get_current_user, get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger
from cift.services.payment_processor import PaymentProcessor

router = APIRouter(prefix="/onboarding", tags=["onboarding"])
payment_processor = PaymentProcessor()


# ============================================================================
# MODELS
# ============================================================================

class KYCProfile(BaseModel):
    """KYC profile model"""
    user_id: str
    status: str  # 'incomplete', 'pending', 'approved', 'rejected'

    # Personal info
    first_name: str | None = None
    last_name: str | None = None
    middle_name: str | None = None
    date_of_birth: str | None = None
    ssn_last_four: str | None = None
    phone_number: str | None = None

    # Address
    street_address: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    country: str | None = None

    # Employment
    employment_status: str | None = None
    employer_name: str | None = None
    occupation: str | None = None
    annual_income: str | None = None
    net_worth: str | None = None

    # Trading experience
    trading_experience: str | None = None
    investment_objectives: list[str] | None = None
    risk_tolerance: str | None = None

    # Documents
    identity_document_uploaded: bool = False
    address_proof_uploaded: bool = False

    # Agreements
    terms_accepted: bool = False
    privacy_accepted: bool = False
    risk_disclosure_accepted: bool = False

    created_at: datetime | None = None
    updated_at: datetime | None = None
    reviewed_at: datetime | None = None
    reviewer_notes: str | None = None


class UpdateKYCRequest(BaseModel):
    """Update KYC profile request"""
    # Personal info
    first_name: str | None = None
    last_name: str | None = None
    middle_name: str | None = None
    date_of_birth: str | None = None
    ssn: str | None = None
    phone_number: str | None = None

    # Address
    street_address: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    country: str | None = None

    # Employment
    employment_status: str | None = None
    employer_name: str | None = None
    occupation: str | None = None
    annual_income: str | None = None
    net_worth: str | None = None

    # Trading experience
    trading_experience: str | None = None
    investment_objectives: list[str] | None = None
    risk_tolerance: str | None = None


class AcceptAgreementsRequest(BaseModel):
    """Accept legal agreements"""
    terms_accepted: bool
    privacy_accepted: bool
    risk_disclosure_accepted: bool


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: str
    document_type: str
    file_name: str
    uploaded_at: datetime


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/profile")
async def get_kyc_profile(
    user_id: UUID = Depends(get_current_user),
):
    """Get KYC profile from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                user_id::text,
                status,
                first_name,
                last_name,
                middle_name,
                date_of_birth,
                ssn_last_four,
                phone_number,
                street_address,
                city,
                state,
                zip_code,
                country,
                employment_status,
                employer_name,
                occupation,
                annual_income,
                net_worth,
                trading_experience,
                investment_objectives,
                risk_tolerance,
                identity_document_uploaded,
                address_proof_uploaded,
                terms_accepted,
                privacy_accepted,
                risk_disclosure_accepted,
                created_at,
                updated_at,
                reviewed_at,
                reviewer_notes
            FROM kyc_profiles
            WHERE user_id = $1
            """,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="KYC profile not found")

        return KYCProfile(
            user_id=row['user_id'],
            status=row['status'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            middle_name=row['middle_name'],
            date_of_birth=row['date_of_birth'],
            ssn_last_four=row['ssn_last_four'],
            phone_number=row['phone_number'],
            street_address=row['street_address'],
            city=row['city'],
            state=row['state'],
            zip_code=row['zip_code'],
            country=row['country'],
            employment_status=row['employment_status'],
            employer_name=row['employer_name'],
            occupation=row['occupation'],
            annual_income=row['annual_income'],
            net_worth=row['net_worth'],
            trading_experience=row['trading_experience'],
            investment_objectives=row['investment_objectives'],
            risk_tolerance=row['risk_tolerance'],
            identity_document_uploaded=row['identity_document_uploaded'],
            address_proof_uploaded=row['address_proof_uploaded'],
            terms_accepted=row['terms_accepted'],
            privacy_accepted=row['privacy_accepted'],
            risk_disclosure_accepted=row['risk_disclosure_accepted'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            reviewed_at=row['reviewed_at'],
            reviewer_notes=row['reviewer_notes'],
        )


@router.post("/profile")
async def create_kyc_profile(
    user_id: UUID = Depends(get_current_user),
):
    """Create initial KYC profile in database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Check if profile already exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM kyc_profiles WHERE user_id = $1)",
            user_id,
        )

        if exists:
            raise HTTPException(status_code=400, detail="KYC profile already exists")

        row = await conn.fetchrow(
            """
            INSERT INTO kyc_profiles (user_id, status)
            VALUES ($1, 'incomplete')
            RETURNING user_id::text, status, created_at
            """,
            user_id,
        )

        return {
            "user_id": row['user_id'],
            "status": row['status'],
            "created_at": row['created_at'],
        }


@router.put("/profile")
async def update_kyc_profile(
    request: UpdateKYCRequest,
    user_id: UUID = Depends(get_current_user),
):
    """Update KYC profile in database"""
    pool = await get_postgres_pool()

    # Build dynamic update query
    updates = []
    params = [user_id]
    param_count = 2

    update_fields = {
        'first_name': request.first_name,
        'last_name': request.last_name,
        'middle_name': request.middle_name,
        'date_of_birth': request.date_of_birth,
        'phone_number': request.phone_number,
        'street_address': request.street_address,
        'city': request.city,
        'state': request.state,
        'zip_code': request.zip_code,
        'country': request.country,
        'employment_status': request.employment_status,
        'employer_name': request.employer_name,
        'occupation': request.occupation,
        'annual_income': request.annual_income,
        'net_worth': request.net_worth,
        'trading_experience': request.trading_experience,
        'investment_objectives': request.investment_objectives,
        'risk_tolerance': request.risk_tolerance,
    }

    # Handle SSN separately (encrypt last 4 digits only)
    if request.ssn:
        updates.append(f"ssn_encrypted = ${param_count}")
        params.append(request.ssn)
        param_count += 1

        updates.append(f"ssn_last_four = ${param_count}")
        params.append(request.ssn[-4:])
        param_count += 1

    for field, value in update_fields.items():
        if value is not None:
            updates.append(f"{field} = ${param_count}")
            params.append(value)
            param_count += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates.append(f"updated_at = ${param_count}")
    params.append(datetime.utcnow())

    query = f"""
        UPDATE kyc_profiles
        SET {', '.join(updates)}
        WHERE user_id = $1
        RETURNING user_id::text, status, updated_at
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)

        if not row:
            raise HTTPException(status_code=404, detail="KYC profile not found")

        return {
            "user_id": row['user_id'],
            "status": row['status'],
            "updated_at": row['updated_at'],
        }


@router.post("/submit")
async def submit_application(
    user_id: UUID = Depends(get_current_user),
):
    """
    Submit KYC application to Alpaca for Brokerage Account creation.
    Requires KYC profile to be filled out.
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # 1. Get User Email
        email = await conn.fetchval("SELECT email FROM users WHERE id = $1", user_id)

        # 2. Get KYC Profile
        profile = await conn.fetchrow(
            """
            SELECT * FROM kyc_profiles WHERE user_id = $1
            """,
            user_id
        )

        if not profile:
            raise HTTPException(status_code=400, detail="KYC profile incomplete")

        # Validate required fields
        required = ['first_name', 'last_name', 'street_address', 'city', 'state', 'zip_code', 'date_of_birth', 'ssn_encrypted']
        for field in required:
            if not profile[field]:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # 3. Prepare Data for Alpaca
        user_details = {
            'email_address': email,
            'first_name': profile['first_name'],
            'last_name': profile['last_name'],
            'phone_number': profile['phone_number'],
            'street_address': profile['street_address'],
            'city': profile['city'],
            'state': profile['state'],
            'postal_code': profile['zip_code'],
            'country': profile['country'] or 'USA',
            'date_of_birth': profile['date_of_birth'], # Ensure YYYY-MM-DD format in frontend
            'tax_id': profile['ssn_encrypted'], # Assuming stored raw for now (see note)
            'tax_id_type': 'USA_SSN'
        }

        try:
            # 4. Create Account via Processor
            account = await payment_processor.create_brokerage_account(user_details)
            alpaca_id = account.get('id')
            status_val = account.get('status')

            # 5. Update User and Profile
            await conn.execute(
                "UPDATE users SET alpaca_account_id = $1 WHERE id = $2",
                alpaca_id, user_id
            )

            await conn.execute(
                "UPDATE kyc_profiles SET status = 'pending', submitted_at = NOW() WHERE user_id = $1",
                user_id
            )

            return {
                "success": True,
                "account_id": alpaca_id,
                "status": status_val,
                "message": "Application submitted to Alpaca"
            }

        except Exception as e:
            logger.error(f"Alpaca submission failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Brokerage account creation failed: {str(e)}")


@router.post("/documents/{document_type}")
async def upload_document(
    document_type: str,
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user),
):
    """Upload identity document to database"""
    if document_type not in ['identity', 'address_proof', 'other']:
        raise HTTPException(status_code=400, detail="Invalid document type")

    # Read file content
    content = await file.read()

    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Store document
        row = await conn.fetchrow(
            """
            INSERT INTO kyc_documents (
                user_id, document_type, file_name, file_size,
                file_content, mime_type, status
            ) VALUES ($1, $2, $3, $4, $5, $6, 'pending')
            RETURNING id::text, document_type, file_name, uploaded_at
            """,
            user_id,
            document_type,
            file.filename,
            len(content),
            content,
            file.content_type,
        )

        # Update KYC profile flags
        if document_type == 'identity':
            await conn.execute(
                "UPDATE kyc_profiles SET identity_document_uploaded = true WHERE user_id = $1",
                user_id,
            )
        elif document_type == 'address_proof':
            await conn.execute(
                "UPDATE kyc_profiles SET address_proof_uploaded = true WHERE user_id = $1",
                user_id,
            )

        # Trigger document verification (async)
        try:
            import asyncio

            from cift.services.kyc_verification import verify_document_async

            # Run verification in background
            asyncio.create_task(verify_document_async(UUID(row['id']), document_type))
            logger.info(f"Started background verification for document {row['id']}")
        except Exception as e:
            logger.warning(f"Failed to start document verification: {e}")

        return DocumentUploadResponse(
            document_id=row['id'],
            document_type=row['document_type'],
            file_name=row['file_name'],
            uploaded_at=row['uploaded_at'],
        )


@router.get("/documents")
async def get_documents(
    user_id: UUID = Depends(get_current_user),
):
    """Get user's uploaded documents from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id::text,
                document_type,
                file_name,
                file_size,
                mime_type,
                status,
                uploaded_at,
                verified_at
            FROM kyc_documents
            WHERE user_id = $1
            ORDER BY uploaded_at DESC
            """,
            user_id,
        )

        return [
            {
                "document_id": row['id'],
                "document_type": row['document_type'],
                "file_name": row['file_name'],
                "file_size": row['file_size'],
                "mime_type": row['mime_type'],
                "status": row['status'],
                "uploaded_at": row['uploaded_at'],
                "verified_at": row['verified_at'],
            }
            for row in rows
        ]


@router.post("/documents/{document_id}/verify")
async def verify_document(
    document_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """Manually trigger document verification"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Check document belongs to user
        doc = await conn.fetchrow(
            "SELECT document_type FROM kyc_documents WHERE id = $1 AND user_id = $2",
            document_id, user_id
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

    try:
        from cift.services.kyc_verification import verify_document_async

        # Run verification
        result = await verify_document_async(document_id, doc['document_type'])

        return {
            "document_id": str(document_id),
            "verification_status": result.verification_status,
            "confidence_score": result.confidence_score,
            "risk_flags": result.risk_flags,
            "processing_time_ms": result.processing_time_ms,
        }

    except Exception as e:
        logger.error(f"Document verification failed: {e}")
        raise HTTPException(status_code=500, detail="Verification failed")


@router.get("/documents/{document_id}/status")
async def get_document_verification_status(
    document_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get document verification status and details"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        doc = await conn.fetchrow("""
            SELECT
                id, document_type, status, verification_details,
                uploaded_at, verified_at
            FROM kyc_documents
            WHERE id = $1 AND user_id = $2
        """, document_id, user_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        verification_details = {}
        if doc['verification_details']:
            try:
                verification_details = json.loads(doc['verification_details'])
            except:
                pass

        return {
            "document_id": str(document_id),
            "document_type": doc['document_type'],
            "status": doc['status'],
            "uploaded_at": doc['uploaded_at'],
            "verified_at": doc['verified_at'],
            "verification_details": verification_details,
        }


@router.post("/verify-identity")
async def verify_user_identity(
    user_id: UUID = Depends(get_current_user_id),
):
    """Trigger complete identity verification for user"""
    try:
        from cift.services.kyc_verification import verify_identity_async

        # Run identity verification
        result = await verify_identity_async(user_id)

        return {
            "user_id": str(user_id),
            "verification_status": result.verification_status,
            "overall_score": result.overall_score,
            "compliance_flags": result.compliance_flags,
            "document_checks": result.document_checks,
            "identity_checks": result.identity_checks,
        }

    except Exception as e:
        logger.error(f"Identity verification failed: {e}")
        raise HTTPException(status_code=500, detail="Identity verification failed")


@router.get("/verification-status")
async def get_verification_status(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get overall verification status for user"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Get KYC profile
        kyc_profile = await conn.fetchrow("""
            SELECT
                status, verification_score, verification_details,
                identity_document_uploaded, address_proof_uploaded,
                phone_verified, email_verified, created_at, verified_at
            FROM kyc_profiles
            WHERE user_id = $1
        """, user_id)

        # Get document statuses
        documents = await conn.fetch("""
            SELECT document_type, status, verified_at
            FROM kyc_documents
            WHERE user_id = $1
            ORDER BY uploaded_at DESC
        """, user_id)

        verification_details = {}
        if kyc_profile and kyc_profile['verification_details']:
            try:
                verification_details = json.loads(kyc_profile['verification_details'])
            except:
                pass

        return {
            "user_id": str(user_id),
            "overall_status": kyc_profile['status'] if kyc_profile else "not_started",
            "verification_score": kyc_profile['verification_score'] if kyc_profile else 0.0,
            "requirements": {
                "identity_document": kyc_profile['identity_document_uploaded'] if kyc_profile else False,
                "address_proof": kyc_profile['address_proof_uploaded'] if kyc_profile else False,
                "phone_verified": kyc_profile['phone_verified'] if kyc_profile else False,
                "email_verified": kyc_profile['email_verified'] if kyc_profile else False,
            },
            "documents": [
                {
                    "type": doc['document_type'],
                    "status": doc['status'],
                    "verified_at": doc['verified_at'],
                }
                for doc in documents
            ],
            "verification_details": verification_details,
            "created_at": kyc_profile['created_at'] if kyc_profile else None,
            "verified_at": kyc_profile['verified_at'] if kyc_profile else None,
        }


@router.post("/agreements")
async def accept_agreements(
    request: AcceptAgreementsRequest,
    user_id: UUID = Depends(get_current_user),
):
    """Accept legal agreements in database"""
    pool = await get_postgres_pool()

    if not all([request.terms_accepted, request.privacy_accepted, request.risk_disclosure_accepted]):
        raise HTTPException(status_code=400, detail="All agreements must be accepted")

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE kyc_profiles
            SET
                terms_accepted = $1,
                privacy_accepted = $2,
                risk_disclosure_accepted = $3,
                updated_at = $4
            WHERE user_id = $5
            """,
            request.terms_accepted,
            request.privacy_accepted,
            request.risk_disclosure_accepted,
            datetime.utcnow(),
            user_id,
        )

        return {"success": True, "message": "Agreements accepted"}


@router.post("/submit")
async def submit_for_review(
    user_id: UUID = Depends(get_current_user),
):
    """Submit KYC profile for review"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Verify profile is complete
        profile = await conn.fetchrow(
            """
            SELECT
                first_name, last_name, date_of_birth, ssn_last_four,
                street_address, city, state, zip_code,
                employment_status, trading_experience,
                identity_document_uploaded, terms_accepted,
                privacy_accepted, risk_disclosure_accepted
            FROM kyc_profiles
            WHERE user_id = $1
            """,
            user_id,
        )

        if not profile:
            raise HTTPException(status_code=404, detail="KYC profile not found")

        # Check required fields
        required_fields = [
            'first_name', 'last_name', 'date_of_birth', 'ssn_last_four',
            'street_address', 'city', 'state', 'zip_code',
            'employment_status', 'trading_experience',
        ]

        missing = [f for f in required_fields if not profile[f]]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing)}"
            )

        if not profile['identity_document_uploaded']:
            raise HTTPException(status_code=400, detail="Identity document required")

        if not all([
            profile['terms_accepted'],
            profile['privacy_accepted'],
            profile['risk_disclosure_accepted'],
        ]):
            raise HTTPException(status_code=400, detail="All agreements must be accepted")

        # Update status to pending
        await conn.execute(
            """
            UPDATE kyc_profiles
            SET status = 'pending', updated_at = $1
            WHERE user_id = $2
            """,
            datetime.utcnow(),
            user_id,
        )

        logger.info(f"KYC profile submitted for review: user_id={user_id}")

        # TODO: Trigger automated verification checks
        # TODO: Send to compliance team queue

        return {
            "success": True,
            "status": "pending",
            "message": "Profile submitted for review",
        }
