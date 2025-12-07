"""
CIFT Markets - Authentication API Routes

Endpoints for user registration, login, token refresh, and API key management.
"""

from datetime import datetime
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status
from loguru import logger
from pydantic import BaseModel

from cift.core.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    create_user,
    decode_token,
    get_current_active_user,
    create_api_key_for_user,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    User,
)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# MODELS
# ============================================================================

class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class APIKeyResponse(BaseModel):
    """API key creation response."""
    api_key: str
    key_id: str
    name: str
    expires_at: datetime | None
    message: str = "IMPORTANT: Save this API key securely. It will not be shown again."


class APIKeyCreateRequest(BaseModel):
    """API key creation request."""
    name: str
    scopes: List[str] = []
    expires_in_days: int | None = None


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user account.
    
    Creates user with default paper trading account ($100,000).
    
    **Requirements**:
    - Valid email address
    - Username (3-50 chars, alphanumeric + _-)
    - Password (8+ chars)
    """
    try:
        user_data = await create_user(
            email=request.email,
            username=request.username,
            password=request.password,
            full_name=request.full_name
        )
        
        logger.info(f"User registered: {user_data['email']}")
        
        return User(**user_data)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate and get access tokens.
    
    Returns JWT access token (30 min) and refresh token (7 days).
    
    **Usage**:
    ```bash
    curl -X POST /api/v1/auth/login \\
      -H "Content-Type: application/json" \\
      -d '{"email":"user@example.com","password":"password"}'
    ```
    """
    # Authenticate user
    user = await authenticate_user(request.email, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(user_id=user['id'])
    refresh_token = create_refresh_token(user_id=user['id'])
    
    logger.info(f"User logged in: {user['email']}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800,  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    
    **Usage**:
    ```bash
    curl -X POST /api/v1/auth/refresh \\
      -H "Content-Type: application/json" \\
      -d '{"refresh_token":"your_refresh_token"}'
    ```
    """
    try:
        # Decode refresh token
        token_data = decode_token(request.refresh_token)
        
        # Verify it's a refresh token
        if token_data.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new tokens
        user_id = UUID(token_data.sub)
        access_token = create_access_token(user_id=user_id)
        new_refresh_token = create_refresh_token(user_id=user_id)
        
        logger.info(f"Token refreshed for user: {user_id}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=1800,
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information.
    
    Requires valid JWT token or API key.
    
    **Usage**:
    ```bash
    curl -X GET /api/v1/auth/me \\
      -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
    ```
    """
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout current user.
    
    Note: This is a placeholder. In production, you should:
    1. Add token to blacklist/revocation list
    2. Invalidate refresh tokens
    3. Clear client-side tokens
    """
    logger.info(f"User logged out: {current_user.email}")
    
    return {
        "message": "Successfully logged out",
        "user_id": str(current_user.id)
    }


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Create new API key for authenticated user.
    
    **Important**: The API key is shown only once. Store it securely!
    
    **Usage**:
    ```bash
    curl -X POST /api/v1/auth/api-keys \\
      -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{"name":"Trading Bot Key","expires_in_days":90}'
    ```
    
    **Scopes** (optional):
    - `trading:read` - Read trading data
    - `trading:write` - Execute trades
    - `market_data:read` - Access market data
    - `account:read` - Read account info
    - `account:write` - Modify account
    """
    try:
        from datetime import timedelta
        
        api_key, key_id = await create_api_key_for_user(
            user_id=current_user.id,
            name=request.name,
            scopes=request.scopes,
            expires_in_days=request.expires_in_days
        )
        
        expires_at = None
        if request.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
        
        logger.info(f"API key created for user {current_user.email}: {request.name}")
        
        return APIKeyResponse(
            api_key=api_key,
            key_id=key_id,
            name=request.name,
            expires_at=expires_at
        )
    
    except Exception as e:
        logger.error(f"API key creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/api-keys")
async def list_api_keys(current_user: User = Depends(get_current_active_user)):
    """
    List all API keys for authenticated user.
    
    Returns key metadata only (not the actual keys).
    """
    from cift.core.database import db_manager
    
    query = """
        SELECT id, name, scopes, is_active, expires_at, created_at, last_used_at
        FROM api_keys
        WHERE user_id = $1
        ORDER BY created_at DESC
    """
    
    async with db_manager.pool.acquire() as conn:
        rows = await conn.fetch(query, current_user.id)
    
    return [dict(row) for row in rows]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: UUID,
    current_user: User = Depends(get_current_active_user)
):
    """
    Revoke (deactivate) an API key.
    
    The key will no longer be valid for authentication.
    """
    from cift.core.database import db_manager
    
    query = """
        UPDATE api_keys
        SET is_active = FALSE
        WHERE id = $1 AND user_id = $2
        RETURNING id
    """
    
    async with db_manager.pool.acquire() as conn:
        result = await conn.fetchrow(query, key_id, current_user.id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    logger.info(f"API key revoked: {key_id} by user {current_user.email}")
    
    return {
        "message": "API key revoked successfully",
        "key_id": str(key_id)
    }


# ============================================================================
# PASSWORD MANAGEMENT
# ============================================================================

class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Change user password.
    
    Requires current password for verification.
    """
    from cift.core.auth import verify_password, hash_password
    from cift.core.database import db_manager
    
    # Get current hashed password
    query = "SELECT hashed_password FROM users WHERE id = $1"
    
    async with db_manager.pool.acquire() as conn:
        result = await conn.fetchrow(query, current_user.id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify current password
    if not verify_password(request.current_password, result['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Hash new password
    new_hashed = hash_password(request.new_password)
    
    # Update password
    update_query = """
        UPDATE users
        SET hashed_password = $1, updated_at = CURRENT_TIMESTAMP
        WHERE id = $2
    """
    
    async with db_manager.pool.acquire() as conn:
        await conn.execute(update_query, new_hashed, current_user.id)
    
    logger.info(f"Password changed for user: {current_user.email}")
    
    return {
        "message": "Password changed successfully"
    }


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@router.get("/users", dependencies=[Depends(get_current_active_user)])
async def list_users(
    skip: int = 0,
    limit: int = 100,
):
    """
    List all users (admin only).
    
    TODO: Add superuser check
    """
    from cift.core.database import db_manager
    
    query = """
        SELECT id, email, username, full_name, is_active, is_superuser, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    """
    
    async with db_manager.pool.acquire() as conn:
        rows = await conn.fetch(query, limit, skip)
    
    return [User(**dict(row)) for row in rows]
