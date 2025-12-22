"""
CIFT Markets - Chart Templates API Routes

Endpoints for saving/loading chart configurations (templates).
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool

router = APIRouter(prefix="/chart-templates", tags=["Chart Templates"])


# ============================================================================
# MODELS
# ============================================================================


class ChartTemplateConfig(BaseModel):
    """Chart configuration structure"""

    symbol: str
    timeframe: str
    chartType: str = "candlestick"
    indicators: list[dict] = []
    viewMode: str = "single"
    multiLayout: str | None = None
    multiTimeframes: list[str] | None = None
    drawingIds: list[str] | None = None


class ChartTemplateCreate(BaseModel):
    """Request model for creating chart template"""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    config: ChartTemplateConfig
    is_default: bool = False


class ChartTemplateUpdate(BaseModel):
    """Request model for updating chart template"""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    config: ChartTemplateConfig | None = None
    is_default: bool | None = None


class ChartTemplateResponse(BaseModel):
    """Response model for chart template"""

    id: str
    user_id: str
    name: str
    description: str | None
    config: dict
    is_default: bool
    created_at: datetime
    updated_at: datetime


# ============================================================================
# ROUTES
# ============================================================================


@router.get("", response_model=list[ChartTemplateResponse])
async def get_templates(user_id: UUID = Depends(get_current_user_id)):
    """
    Get all chart templates for the authenticated user.
    """
    pool = await get_postgres_pool()

    query = """
        SELECT id, user_id, name, description, config, is_default, created_at, updated_at
        FROM chart_templates
        WHERE user_id = $1
        ORDER BY is_default DESC, created_at DESC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, user_id)

        return [
            ChartTemplateResponse(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                name=row["name"],
                description=row["description"],
                config=row["config"],
                is_default=row["is_default"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]


@router.get("/{template_id}", response_model=ChartTemplateResponse)
async def get_template(template_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """
    Get a specific chart template by ID.
    """
    pool = await get_postgres_pool()

    query = """
        SELECT id, user_id, name, description, config, is_default, created_at, updated_at
        FROM chart_templates
        WHERE id = $1 AND user_id = $2
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, template_id, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="Template not found")

        return ChartTemplateResponse(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=row["name"],
            description=row["description"],
            config=row["config"],
            is_default=row["is_default"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@router.post("", response_model=ChartTemplateResponse, status_code=201)
async def create_template(
    template: ChartTemplateCreate, user_id: UUID = Depends(get_current_user_id)
):
    """
    Create a new chart template.
    """
    pool = await get_postgres_pool()

    # If setting as default, unset other defaults first
    async with pool.acquire() as conn:
        if template.is_default:
            await conn.execute(
                "UPDATE chart_templates SET is_default = false WHERE user_id = $1", user_id
            )

        query = """
            INSERT INTO chart_templates (user_id, name, description, config, is_default)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, user_id, name, description, config, is_default, created_at, updated_at
        """

        try:
            row = await conn.fetchrow(
                query,
                user_id,
                template.name,
                template.description,
                template.config.dict(),
                template.is_default,
            )

            logger.info(f"Created chart template: {template.name} for user {user_id}")

            return ChartTemplateResponse(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                name=row["name"],
                description=row["description"],
                config=row["config"],
                is_default=row["is_default"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except Exception as e:
            if "unique_user_template_name" in str(e):
                raise HTTPException(status_code=400, detail="Template name already exists") from e
            raise HTTPException(
                status_code=500, detail=f"Failed to create template: {str(e)}"
            ) from e


@router.patch("/{template_id}", response_model=ChartTemplateResponse)
async def update_template(
    template_id: UUID,
    template_update: ChartTemplateUpdate,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Update an existing chart template.
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Check if template exists and belongs to user
        existing = await conn.fetchrow(
            "SELECT id FROM chart_templates WHERE id = $1 AND user_id = $2", template_id, user_id
        )

        if not existing:
            raise HTTPException(status_code=404, detail="Template not found")

        # If setting as default, unset other defaults first
        if template_update.is_default:
            await conn.execute(
                "UPDATE chart_templates SET is_default = false WHERE user_id = $1 AND id != $2",
                user_id,
                template_id,
            )

        # Build dynamic update query
        updates = []
        params = [template_id, user_id]
        param_idx = 3

        if template_update.name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(template_update.name)
            param_idx += 1

        if template_update.description is not None:
            updates.append(f"description = ${param_idx}")
            params.append(template_update.description)
            param_idx += 1

        if template_update.config is not None:
            updates.append(f"config = ${param_idx}")
            params.append(template_update.config.dict())
            param_idx += 1

        if template_update.is_default is not None:
            updates.append(f"is_default = ${param_idx}")
            params.append(template_update.is_default)
            param_idx += 1

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        query = f"""
            UPDATE chart_templates
            SET {', '.join(updates)}
            WHERE id = $1 AND user_id = $2
            RETURNING id, user_id, name, description, config, is_default, created_at, updated_at
        """

        try:
            row = await conn.fetchrow(query, *params)

            logger.info(f"Updated chart template: {template_id} for user {user_id}")

            return ChartTemplateResponse(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                name=row["name"],
                description=row["description"],
                config=row["config"],
                is_default=row["is_default"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except Exception as e:
            if "unique_user_template_name" in str(e):
                raise HTTPException(status_code=400, detail="Template name already exists") from e
            raise HTTPException(
                status_code=500, detail=f"Failed to update template: {str(e)}"
            ) from e


@router.delete("/{template_id}")
async def delete_template(template_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """
    Delete a chart template.
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chart_templates WHERE id = $1 AND user_id = $2", template_id, user_id
        )

        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Template not found")

        logger.info(f"Deleted chart template: {template_id} for user {user_id}")

        return {"message": "Template deleted successfully"}


@router.get("/default/get", response_model=ChartTemplateResponse | None)
async def get_default_template(user_id: UUID = Depends(get_current_user_id)):
    """
    Get the user's default chart template.
    """
    pool = await get_postgres_pool()

    query = """
        SELECT id, user_id, name, description, config, is_default, created_at, updated_at
        FROM chart_templates
        WHERE user_id = $1 AND is_default = true
        LIMIT 1
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, user_id)

        if not row:
            return None

        return ChartTemplateResponse(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=row["name"],
            description=row["description"],
            config=row["config"],
            is_default=row["is_default"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
