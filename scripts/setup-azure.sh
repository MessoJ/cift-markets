#!/bin/bash
set -e

echo "🚀 Starting CIFT Markets Azure Setup..."

# 1. Update System (skip upgrade to avoid interactive prompts/long wait)
echo "📦 Updating system packages..."
sudo apt-get update

# 2. Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed successfully."
else
    echo "✅ Docker is already installed."
fi

# 3. Install Docker Compose Plugin
echo "🐙 Installing Docker Compose..."
sudo apt-get install -y docker-compose-plugin

echo "✅ Setup Complete! Please logout and login again to use Docker without sudo."
