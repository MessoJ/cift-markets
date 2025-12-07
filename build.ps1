#!/usr/bin/env pwsh
<#
.SYNOPSIS
    CIFT Markets - Smart Docker Build Script
    
.DESCRIPTION
    Builds Docker containers with intelligent caching.
    - Frontend: ~2-4 minutes (deps cached unless package.json changes)
    - API Dev:  ~1-2 minutes (Python only, no Rust)
    - API Prod: ~5-10 minutes (with Rust, but cached on subsequent builds)

.PARAMETER Target
    What to build: 'frontend', 'api', 'all'
    
.PARAMETER Mode
    Build mode: 'dev' (fast), 'prod' (with Rust optimization)
    
.PARAMETER NoCache
    Force rebuild without using cache

.EXAMPLE
    .\build.ps1 frontend
    .\build.ps1 api -Mode prod
    .\build.ps1 all -NoCache
#>

param(
    [Parameter(Position=0)]
    [ValidateSet('frontend', 'api', 'all')]
    [string]$Target = 'all',
    
    [ValidateSet('dev', 'prod')]
    [string]$Mode = 'dev',
    
    [switch]$NoCache
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Enable BuildKit for better caching
$env:DOCKER_BUILDKIT = 1
$env:COMPOSE_DOCKER_CLI_BUILD = 1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CIFT Markets Docker Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Target: $Target | Mode: $Mode | Cache: $(if ($NoCache) { 'OFF' } else { 'ON' })" -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date

function Build-Frontend {
    Write-Host "`n[Frontend] Building SolidJS app..." -ForegroundColor Green
    $cacheFlag = if ($NoCache) { "--no-cache" } else { "" }
    
    $cmd = "docker-compose build $cacheFlag frontend"
    Write-Host "Running: $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[Frontend] Build successful!" -ForegroundColor Green
    } else {
        Write-Host "[Frontend] Build FAILED!" -ForegroundColor Red
        exit 1
    }
}

function Build-Api {
    param([string]$BuildMode)
    
    $dockerfile = if ($BuildMode -eq 'prod') { 'Dockerfile.prod' } else { 'Dockerfile.dev' }
    Write-Host "`n[API] Building with $dockerfile..." -ForegroundColor Green
    
    $cacheFlag = if ($NoCache) { "--no-cache" } else { "" }
    $env:API_DOCKERFILE = $dockerfile
    
    $cmd = "docker-compose build $cacheFlag api"
    Write-Host "Running: $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[API] Build successful!" -ForegroundColor Green
    } else {
        Write-Host "[API] Build FAILED!" -ForegroundColor Red
        exit 1
    }
}

# Execute builds
switch ($Target) {
    'frontend' { Build-Frontend }
    'api' { Build-Api -BuildMode $Mode }
    'all' {
        Build-Frontend
        Build-Api -BuildMode $Mode
    }
}

# Timing
$elapsed = (Get-Date) - $startTime
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Build completed in $($elapsed.Minutes)m $($elapsed.Seconds)s" -ForegroundColor Green
Write-Host ""

# Restart hint
Write-Host "To apply changes, run:" -ForegroundColor Yellow
Write-Host "  docker-compose up -d $Target" -ForegroundColor White
Write-Host ""
