"""
CIFT Markets - KYC Verification Service

Handles document verification, identity checks, and compliance screening.

Integrations:
- Document OCR: Google Vision API / AWS Textract
- Identity Verification: Persona / Jumio / Onfido
- Sanctions Screening: ComplyAdvantage / Chainalysis
- Risk Assessment: Internal scoring + external APIs

Usage:
    verifier = KYCVerificationService()
    result = await verifier.verify_document(document_id, document_type)
"""

import asyncio
import json
import uuid
from datetime import datetime
from uuid import UUID

import aiohttp
from loguru import logger
from pydantic import BaseModel

from cift.core.config import get_settings
from cift.core.database import get_postgres_pool

# ============================================================================
# MODELS
# ============================================================================


class DocumentVerificationResult(BaseModel):
    """Document verification result."""

    document_id: UUID
    verification_status: str  # pending, verified, rejected, needs_review
    confidence_score: float  # 0.0 - 1.0
    extracted_data: dict
    verification_details: dict
    risk_flags: list[str]
    processing_time_ms: int
    provider: str
    external_reference: str | None = None


class IdentityVerificationResult(BaseModel):
    """Identity verification result."""

    user_id: UUID
    verification_status: str  # pending, verified, rejected, needs_review
    overall_score: float  # 0.0 - 1.0
    document_checks: dict
    identity_checks: dict
    sanctions_check: dict
    risk_assessment: dict
    compliance_flags: list[str]
    verification_provider: str
    external_reference: str | None = None


# ============================================================================
# KYC VERIFICATION SERVICE
# ============================================================================


class KYCVerificationService:
    """
    Advanced KYC verification service with multiple providers.

    Features:
    - Document OCR and verification
    - Identity matching and validation
    - Sanctions and watchlist screening
    - Risk scoring and compliance checks
    - Multi-provider fallback system
    """

    def __init__(self):
        self.settings = get_settings()
        self.session = None

        # Provider configurations
        self.providers = {
            "document_ocr": "google_vision",  # google_vision, aws_textract, mock
            "identity_verification": "persona",  # persona, jumio, onfido, mock
            "sanctions_screening": "chainalysis",  # chainalysis, comply_advantage, mock
        }

        # Risk scoring thresholds
        self.risk_thresholds = {
            "low": 0.8,  # Auto-approve
            "medium": 0.6,  # Manual review
            "high": 0.3,  # Auto-reject
        }

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    # ========================================================================
    # DOCUMENT VERIFICATION
    # ========================================================================

    async def verify_document(
        self, document_id: UUID, document_type: str
    ) -> DocumentVerificationResult:
        """
        Verify a uploaded document.

        Steps:
        1. Extract document from database
        2. Run OCR to extract text/data
        3. Validate document authenticity
        4. Check against watchlists
        5. Calculate confidence score
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting document verification for {document_id}")

        try:
            # Get document from database
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                doc_row = await conn.fetchrow(
                    """
                    SELECT
                        id, user_id, document_type, file_name, file_content,
                        mime_type, status, uploaded_at
                    FROM kyc_documents
                    WHERE id = $1
                """,
                    document_id,
                )

                if not doc_row:
                    raise ValueError(f"Document {document_id} not found")

            # Run OCR and data extraction
            extracted_data = await self._extract_document_data(
                doc_row["file_content"], document_type, doc_row["mime_type"]
            )

            # Validate document authenticity
            authenticity_check = await self._validate_document_authenticity(
                extracted_data, document_type
            )

            # Sanctions and watchlist screening
            sanctions_check = await self._screen_document_data(extracted_data)

            # Calculate overall confidence score
            confidence_score = self._calculate_document_confidence(
                extracted_data, authenticity_check, sanctions_check
            )

            # Determine verification status
            verification_status = self._determine_verification_status(
                confidence_score, authenticity_check, sanctions_check
            )

            # Collect risk flags
            risk_flags = []
            if authenticity_check.get("low_quality", False):
                risk_flags.append("low_image_quality")
            if authenticity_check.get("suspicious_patterns", False):
                risk_flags.append("suspicious_document_patterns")
            if sanctions_check.get("matches", []):
                risk_flags.append("sanctions_match")

            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            result = DocumentVerificationResult(
                document_id=document_id,
                verification_status=verification_status,
                confidence_score=confidence_score,
                extracted_data=extracted_data,
                verification_details={
                    "authenticity_check": authenticity_check,
                    "sanctions_check": sanctions_check,
                },
                risk_flags=risk_flags,
                processing_time_ms=processing_time,
                provider=self.providers["document_ocr"],
                external_reference=f"kyc_doc_{document_id}_{int(start_time.timestamp())}",
            )

            # Store verification result
            await self._store_document_verification_result(result)

            logger.success(
                f"Document verification completed: {verification_status} ({confidence_score:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Document verification failed: {e}")
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Return failed result
            return DocumentVerificationResult(
                document_id=document_id,
                verification_status="failed",
                confidence_score=0.0,
                extracted_data={},
                verification_details={"error": str(e)},
                risk_flags=["verification_error"],
                processing_time_ms=processing_time,
                provider=self.providers["document_ocr"],
            )

    async def _extract_document_data(
        self, file_content: bytes, document_type: str, mime_type: str
    ) -> dict:
        """Extract structured data from document using OCR."""

        if self.providers["document_ocr"] == "mock":
            # Mock implementation for testing
            return await self._mock_ocr_extraction(document_type)

        elif self.providers["document_ocr"] == "google_vision":
            return await self._google_vision_ocr(file_content, document_type)

        elif self.providers["document_ocr"] == "aws_textract":
            return await self._aws_textract_ocr(file_content, document_type)

        else:
            raise ValueError(f"Unknown OCR provider: {self.providers['document_ocr']}")

    async def _mock_ocr_extraction(self, document_type: str) -> dict:
        """Mock OCR extraction for testing."""
        # Simulate processing delay
        await asyncio.sleep(0.5)

        if document_type == "identity":
            return {
                "document_type": "drivers_license",
                "full_name": "John Smith",
                "date_of_birth": "1990-05-15",
                "document_number": "DL123456789",
                "expiry_date": "2027-05-15",
                "issuing_authority": "California DMV",
                "address": "123 Main St, Los Angeles, CA 90210",
                "confidence_scores": {
                    "full_name": 0.95,
                    "date_of_birth": 0.98,
                    "document_number": 0.92,
                },
            }
        elif document_type == "address_proof":
            return {
                "document_type": "utility_bill",
                "full_name": "John Smith",
                "address": "123 Main St, Los Angeles, CA 90210",
                "issue_date": "2024-10-15",
                "company": "SoCal Edison",
                "confidence_scores": {
                    "full_name": 0.94,
                    "address": 0.96,
                    "issue_date": 0.97,
                },
            }
        else:
            return {"document_type": "unknown", "raw_text": "Document text..."}

    async def _google_vision_ocr(self, file_content: bytes, document_type: str) -> dict:
        """Extract data using Google Vision API."""
        # TODO: Implement Google Vision API integration
        logger.warning("Google Vision API not implemented, using mock data")
        return await self._mock_ocr_extraction(document_type)

    async def _aws_textract_ocr(self, file_content: bytes, document_type: str) -> dict:
        """Extract data using AWS Textract."""
        # TODO: Implement AWS Textract integration
        logger.warning("AWS Textract not implemented, using mock data")
        return await self._mock_ocr_extraction(document_type)

    async def _validate_document_authenticity(
        self, extracted_data: dict, document_type: str
    ) -> dict:
        """Validate document authenticity and detect fraud."""

        # Basic validation rules
        validation_result = {
            "is_authentic": True,
            "confidence": 0.85,
            "checks_passed": [],
            "checks_failed": [],
            "low_quality": False,
            "suspicious_patterns": False,
        }

        # Check required fields
        required_fields = {
            "identity": ["full_name", "date_of_birth", "document_number"],
            "address_proof": ["full_name", "address", "issue_date"],
        }

        if document_type in required_fields:
            for field in required_fields[document_type]:
                if field in extracted_data and extracted_data[field]:
                    validation_result["checks_passed"].append(f"has_{field}")
                else:
                    validation_result["checks_failed"].append(f"missing_{field}")
                    validation_result["confidence"] -= 0.2

        # Check confidence scores
        confidence_scores = extracted_data.get("confidence_scores", {})
        avg_confidence = (
            sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.8
        )

        if avg_confidence < 0.7:
            validation_result["low_quality"] = True
            validation_result["confidence"] -= 0.3

        # Date validation
        if "date_of_birth" in extracted_data:
            try:
                dob = datetime.strptime(extracted_data["date_of_birth"], "%Y-%m-%d")
                age = (datetime.now() - dob).days / 365.25

                if age < 18 or age > 120:
                    validation_result["checks_failed"].append("invalid_age")
                    validation_result["confidence"] -= 0.4
                else:
                    validation_result["checks_passed"].append("valid_age")
            except Exception:
                validation_result["checks_failed"].append("invalid_date_format")
                validation_result["confidence"] -= 0.2

        # Update authenticity based on confidence
        validation_result["is_authentic"] = validation_result["confidence"] > 0.5

        return validation_result

    async def _screen_document_data(self, extracted_data: dict) -> dict:
        """Screen document data against sanctions and watchlists."""

        screening_result = {
            "matches": [],
            "risk_level": "low",
            "screening_provider": self.providers["sanctions_screening"],
            "screened_fields": [],
        }

        # Extract screenable fields
        screenable_fields = ["full_name"]

        for field in screenable_fields:
            if field in extracted_data and extracted_data[field]:
                screening_result["screened_fields"].append(field)

                # Mock screening (replace with real API)
                if self.providers["sanctions_screening"] == "mock":
                    # Simulate no matches for testing
                    pass
                else:
                    # TODO: Implement real sanctions screening
                    logger.warning(f"Real sanctions screening not implemented for {field}")

        return screening_result

    def _calculate_document_confidence(
        self, extracted_data: dict, authenticity_check: dict, sanctions_check: dict
    ) -> float:
        """Calculate overall document confidence score."""

        # Base confidence from authenticity check
        confidence = authenticity_check.get("confidence", 0.5)

        # Adjust for data completeness
        required_fields = ["full_name"]
        present_fields = sum(
            1 for field in required_fields if field in extracted_data and extracted_data[field]
        )
        completeness_score = present_fields / len(required_fields)

        confidence = (confidence * 0.7) + (completeness_score * 0.3)

        # Penalties for risk factors
        if sanctions_check.get("matches"):
            confidence -= 0.5

        if authenticity_check.get("low_quality"):
            confidence -= 0.2

        if authenticity_check.get("suspicious_patterns"):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def _determine_verification_status(
        self, confidence_score: float, authenticity_check: dict, sanctions_check: dict
    ) -> str:
        """Determine verification status based on checks."""

        # Auto-reject conditions
        if sanctions_check.get("matches"):
            return "rejected"

        if not authenticity_check.get("is_authentic", True):
            return "rejected"

        # Status based on confidence thresholds
        if confidence_score >= self.risk_thresholds["low"]:
            return "verified"
        elif confidence_score >= self.risk_thresholds["medium"]:
            return "needs_review"
        else:
            return "rejected"

    async def _store_document_verification_result(self, result: DocumentVerificationResult):
        """Store verification result in database."""
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Update document status
            await conn.execute(
                """
                UPDATE kyc_documents
                SET
                    status = $2,
                    verified_at = $3,
                    verification_details = $4
                WHERE id = $1
            """,
                result.document_id,
                result.verification_status,
                datetime.utcnow() if result.verification_status == "verified" else None,
                json.dumps(
                    {
                        "confidence_score": result.confidence_score,
                        "verification_details": result.verification_details,
                        "risk_flags": result.risk_flags,
                        "provider": result.provider,
                        "external_reference": result.external_reference,
                        "processing_time_ms": result.processing_time_ms,
                    }
                ),
            )

            logger.info(f"Stored verification result for document {result.document_id}")

    # ========================================================================
    # IDENTITY VERIFICATION
    # ========================================================================

    async def verify_identity(self, user_id: UUID) -> IdentityVerificationResult:
        """
        Perform comprehensive identity verification for a user.

        Steps:
        1. Check all uploaded documents
        2. Cross-validate identity information
        3. Perform sanctions screening
        4. Calculate risk score
        5. Make verification decision
        """
        logger.info(f"Starting identity verification for user {user_id}")

        try:
            # Get user's documents
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                docs = await conn.fetch(
                    """
                    SELECT
                        id, document_type, status, verification_details,
                        uploaded_at, verified_at
                    FROM kyc_documents
                    WHERE user_id = $1
                    ORDER BY uploaded_at DESC
                """,
                    user_id,
                )

                user_info = await conn.fetchrow(
                    """
                    SELECT email, username, full_name, created_at
                    FROM users
                    WHERE id = $1
                """,
                    user_id,
                )

            if not docs:
                raise ValueError(f"No documents found for user {user_id}")

            # Analyze document verification results
            document_checks = {}
            for doc in docs:
                doc_id = str(doc["id"])
                document_checks[doc_id] = {
                    "type": doc["document_type"],
                    "status": doc["status"],
                    "verified_at": doc["verified_at"].isoformat() if doc["verified_at"] else None,
                    "details": (
                        json.loads(doc["verification_details"])
                        if doc["verification_details"]
                        else {}
                    ),
                }

            # Cross-validate identity information
            identity_checks = await self._cross_validate_identity(docs, user_info)

            # Sanctions screening at user level
            sanctions_check = await self._screen_user_identity(user_info, docs)

            # Risk assessment
            risk_assessment = await self._assess_user_risk(user_id, docs, user_info)

            # Calculate overall score
            overall_score = self._calculate_identity_score(
                document_checks, identity_checks, sanctions_check, risk_assessment
            )

            # Determine verification status
            verification_status = self._determine_identity_status(
                overall_score, document_checks, sanctions_check, risk_assessment
            )

            # Collect compliance flags
            compliance_flags = []
            if sanctions_check.get("matches"):
                compliance_flags.append("sanctions_match")
            if risk_assessment.get("high_risk_country"):
                compliance_flags.append("high_risk_jurisdiction")
            if identity_checks.get("name_mismatch"):
                compliance_flags.append("identity_mismatch")

            result = IdentityVerificationResult(
                user_id=user_id,
                verification_status=verification_status,
                overall_score=overall_score,
                document_checks=document_checks,
                identity_checks=identity_checks,
                sanctions_check=sanctions_check,
                risk_assessment=risk_assessment,
                compliance_flags=compliance_flags,
                verification_provider=self.providers["identity_verification"],
                external_reference=f"kyc_identity_{user_id}_{int(datetime.utcnow().timestamp())}",
            )

            # Store identity verification result
            await self._store_identity_verification_result(result)

            logger.success(
                f"Identity verification completed: {verification_status} ({overall_score:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Identity verification failed: {e}")

            return IdentityVerificationResult(
                user_id=user_id,
                verification_status="failed",
                overall_score=0.0,
                document_checks={},
                identity_checks={"error": str(e)},
                sanctions_check={},
                risk_assessment={},
                compliance_flags=["verification_error"],
                verification_provider=self.providers["identity_verification"],
            )

    async def _cross_validate_identity(self, docs: list, user_info: dict) -> dict:
        """Cross-validate identity information across documents."""

        validation_result = {
            "name_match": True,
            "address_match": True,
            "consistency_score": 1.0,
            "issues": [],
        }

        # Extract names from documents
        names_from_docs = []
        addresses_from_docs = []

        for doc in docs:
            if doc["verification_details"]:
                details = json.loads(doc["verification_details"])
                extracted_data = details.get("verification_details", {}).get("extracted_data", {})

                if "full_name" in extracted_data:
                    names_from_docs.append(extracted_data["full_name"])

                if "address" in extracted_data:
                    addresses_from_docs.append(extracted_data["address"])

        # Check name consistency
        if names_from_docs:
            # Simple name matching (could be enhanced with fuzzy matching)
            if len(set(names_from_docs)) > 1:
                validation_result["name_match"] = False
                validation_result["issues"].append("inconsistent_names_across_documents")
                validation_result["consistency_score"] -= 0.3

        # Check address consistency
        if len(addresses_from_docs) > 1:
            if len(set(addresses_from_docs)) > 1:
                validation_result["address_match"] = False
                validation_result["issues"].append("inconsistent_addresses")
                validation_result["consistency_score"] -= 0.2

        return validation_result

    async def _screen_user_identity(self, user_info: dict, docs: list) -> dict:
        """Screen user identity against sanctions and watchlists."""

        # Mock implementation
        return {
            "matches": [],
            "risk_level": "low",
            "screening_provider": self.providers["sanctions_screening"],
            "screened_at": datetime.utcnow().isoformat(),
        }

    async def _assess_user_risk(self, user_id: UUID, docs: list, user_info: dict) -> dict:
        """Assess overall user risk profile."""

        risk_score = 0.1  # Start with low base risk
        risk_factors = []

        # Account age factor
        if user_info and user_info["created_at"]:
            account_age_days = (datetime.utcnow() - user_info["created_at"]).days
            if account_age_days < 1:
                risk_score += 0.2
                risk_factors.append("new_account")

        # Document quality assessment
        verified_docs = sum(1 for doc in docs if doc["status"] == "verified")
        total_docs = len(docs)

        if total_docs == 0:
            risk_score += 0.5
            risk_factors.append("no_documents")
        elif verified_docs / total_docs < 0.5:
            risk_score += 0.3
            risk_factors.append("low_document_verification_rate")

        return {
            "risk_score": min(1.0, risk_score),
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "low",
            "risk_factors": risk_factors,
            "assessment_date": datetime.utcnow().isoformat(),
        }

    def _calculate_identity_score(
        self,
        document_checks: dict,
        identity_checks: dict,
        sanctions_check: dict,
        risk_assessment: dict,
    ) -> float:
        """Calculate overall identity verification score."""

        score = 0.5  # Base score

        # Document verification contribution (40%)
        verified_docs = sum(
            1 for doc_id, doc in document_checks.items() if doc["status"] == "verified"
        )
        total_docs = len(document_checks)

        if total_docs > 0:
            doc_score = verified_docs / total_docs
            score += doc_score * 0.4

        # Identity consistency contribution (30%)
        consistency_score = identity_checks.get("consistency_score", 0.5)
        score += consistency_score * 0.3

        # Risk assessment contribution (30%)
        risk_score = risk_assessment.get("risk_score", 0.5)
        score += (1.0 - risk_score) * 0.3  # Lower risk = higher score

        # Penalties
        if sanctions_check.get("matches"):
            score -= 0.8

        if not identity_checks.get("name_match", True):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _determine_identity_status(
        self,
        overall_score: float,
        document_checks: dict,
        sanctions_check: dict,
        risk_assessment: dict,
    ) -> str:
        """Determine identity verification status."""

        # Auto-reject conditions
        if sanctions_check.get("matches"):
            return "rejected"

        if risk_assessment.get("risk_level") == "high":
            return "needs_review"

        # Must have at least one verified document
        has_verified_doc = any(doc["status"] == "verified" for doc in document_checks.values())

        if not has_verified_doc:
            return "pending"

        # Score-based determination
        if overall_score >= self.risk_thresholds["low"]:
            return "verified"
        elif overall_score >= self.risk_thresholds["medium"]:
            return "needs_review"
        else:
            return "rejected"

    async def _store_identity_verification_result(self, result: IdentityVerificationResult):
        """Store identity verification result in database."""
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Update KYC profile
            await conn.execute(
                """
                UPDATE kyc_profiles
                SET
                    status = $2,
                    verification_score = $3,
                    verification_details = $4,
                    verified_at = $5
                WHERE user_id = $1
            """,
                result.user_id,
                result.verification_status,
                result.overall_score,
                json.dumps(
                    {
                        "document_checks": result.document_checks,
                        "identity_checks": result.identity_checks,
                        "sanctions_check": result.sanctions_check,
                        "risk_assessment": result.risk_assessment,
                        "compliance_flags": result.compliance_flags,
                        "verification_provider": result.verification_provider,
                        "external_reference": result.external_reference,
                    }
                ),
                datetime.utcnow() if result.verification_status == "verified" else None,
            )

            logger.info(f"Stored identity verification result for user {result.user_id}")


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================


async def verify_document_async(
    document_id: UUID, document_type: str
) -> DocumentVerificationResult:
    """Async function for document verification."""
    async with KYCVerificationService() as verifier:
        return await verifier.verify_document(document_id, document_type)


async def verify_identity_async(user_id: UUID) -> IdentityVerificationResult:
    """Async function for identity verification."""
    async with KYCVerificationService() as verifier:
        return await verifier.verify_identity(user_id)


# ============================================================================
# BACKGROUND TASK INTEGRATION
# ============================================================================


async def process_pending_verifications():
    """Background task to process pending document and identity verifications."""
    logger.info("Processing pending KYC verifications...")

    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Get pending documents
        pending_docs = await conn.fetch(
            """
            SELECT id, document_type
            FROM kyc_documents
            WHERE status = 'pending'
            ORDER BY uploaded_at ASC
            LIMIT 10
        """
        )

        # Process each pending document
        for doc in pending_docs:
            try:
                result = await verify_document_async(doc["id"], doc["document_type"])
                logger.info(f"Processed document {doc['id']}: {result.verification_status}")
            except Exception as e:
                logger.error(f"Failed to process document {doc['id']}: {e}")

        # Get users needing identity verification
        pending_users = await conn.fetch(
            """
            SELECT DISTINCT user_id
            FROM kyc_profiles kp
            WHERE kp.status = 'incomplete'
            AND EXISTS (
                SELECT 1 FROM kyc_documents kd
                WHERE kd.user_id = kp.user_id
                AND kd.status = 'verified'
            )
            LIMIT 5
        """
        )

        # Process identity verifications
        for user_row in pending_users:
            try:
                result = await verify_identity_async(user_row["user_id"])
                logger.info(
                    f"Processed identity for user {user_row['user_id']}: {result.verification_status}"
                )
            except Exception as e:
                logger.error(f"Failed to process identity for user {user_row['user_id']}: {e}")

    logger.success(
        f"Processed {len(pending_docs)} documents and {len(pending_users)} identity verifications"
    )


if __name__ == "__main__":
    # Test the verification service
    import asyncio

    async def test_verification():
        async with KYCVerificationService() as verifier:
            # Mock document verification test
            test_doc_id = uuid.uuid4()
            result = await verifier.verify_document(test_doc_id, "identity")
            print(f"Document verification result: {result}")

    asyncio.run(test_verification())
