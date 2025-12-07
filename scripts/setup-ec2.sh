#!/bin/bash

# CIFT Markets - EC2 Setup Script
# Run this script on your fresh Ubuntu EC2 instance to prepare it for deployment.

set -e

echo "🚀 Starting CIFT Markets Server Setup..."

# 1. Update System
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

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

# 3. Install Docker Compose
echo "🐙 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
    # Alias for convenience if needed, though 'docker compose' is preferred now
    echo "✅ Docker Compose installed."
else
    echo "✅ Docker Compose is already installed."
fi

# 4. Install Git
echo "Git Installing Git..."
sudo apt-get install -y git

# 5. Clone/Setup Repository
echo "📂 Setting up repository..."
REPO_DIR=~/cift-markets
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists. Pulling latest..."
    cd $REPO_DIR
    git pull
else
    echo "Cloning repository..."
    # NOTE: You might need to use HTTPS or set up SSH keys for this to work automatically
    # For public repos:
    # git clone https://github.com/YOUR_USERNAME/cift-markets.git $REPO_DIR
    
    # For private repos, we'll create the directory and you can pull later
    mkdir -p $REPO_DIR
    echo "⚠️  Created directory $REPO_DIR. Please clone your repo into this folder or copy files."
fi

# 6. Create necessary directories for volumes
echo "🗄️ Creating volume directories..."
cd $REPO_DIR
mkdir -p logs
mkdir -p cift
mkdir -p scripts

# 7. Setup Swap (Optional but recommended for smaller instances)
echo "🔄 Checking Swap..."
if [ $(swapon --show | wc -l) -eq 0 ]; then
    echo "Creating 4GB Swap file..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "✅ Swap created."
else
    echo "✅ Swap already exists."
fi

echo "✨ Setup Complete! Please log out and log back in for Docker group changes to take effect."
echo "📝 Next Steps:"
echo "1. Copy your .env file to $REPO_DIR/.env"
echo "2. Run 'docker compose up -d' to start services manually for the first time."
