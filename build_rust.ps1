# Simple Rust core build script
Write-Host "Building Rust core modules..." -ForegroundColor Cyan

# Check Rust
if (!(Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Rust not installed. Install from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Install maturin if needed
if (!(Get-Command maturin -ErrorAction SilentlyContinue)) {
    Write-Host "Installing maturin..." -ForegroundColor Yellow
    pip install maturin
}

# Build
cd rust_core

if ($args[0] -eq "release") {
    Write-Host "Building RELEASE mode..." -ForegroundColor Yellow
    maturin build --release
} else {
    Write-Host "Building DEV mode..." -ForegroundColor Yellow
    maturin develop
}

cd ..

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Build successful!" -ForegroundColor Green
    
    # Test import
    python -c "from cift_core import FastOrderBook; print('✓ Rust modules loaded')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ All checks passed!" -ForegroundColor Green
    }
} else {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}
