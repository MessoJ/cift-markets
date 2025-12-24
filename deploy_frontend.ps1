# Deploy Frontend Fixes to Azure
# This script builds the frontend locally and deploys it to Azure.

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Building frontend..."
cd frontend
npm run build
cd ..

Write-Host "Deploying frontend files to $ServerIP..."

# Copy the build output to the server
# We'll copy the dist folder to the server and then mount it or rebuild the container
# For simplicity/speed, we'll copy the source files we changed and rebuild the container on the server

# 1. Copy ArticleDetailPage.tsx
scp frontend/src/pages/news/ArticleDetailPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/news/ArticleDetailPage.tsx

# 2. Copy TradingPage.tsx
scp frontend/src/pages/trading/TradingPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/trading/TradingPage.tsx

# 3. Copy Auth Pages
scp frontend/src/pages/auth/RegisterPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/auth/RegisterPage.tsx
scp frontend/src/pages/auth/LoginPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/auth/LoginPage.tsx

# 4. Copy Watchlists Page
scp frontend/src/pages/watchlists/WatchlistsPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/watchlists/WatchlistsPage.tsx

# 5. Copy Alerts Page
scp frontend/src/pages/alerts/AlertsPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/alerts/AlertsPage.tsx

# 6. Copy Profile & Settings Pages
scp frontend/src/pages/profile/ProfilePage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/profile/ProfilePage.tsx
scp frontend/src/pages/settings/SettingsPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/settings/SettingsPage.tsx

# 7. Rebuild Frontend Container on Server
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Frontend deployment complete!"
