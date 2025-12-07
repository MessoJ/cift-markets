# 🚀 Deploying CIFT Markets to AWS

This guide explains how to host the CIFT Markets platform on AWS EC2 using Docker Compose and GitHub Actions for automated deployments.

## 📋 Prerequisites

1.  **AWS Account**: You need access to the AWS Console.
2.  **GitHub Repository**: This code must be pushed to a GitHub repository.

---

## 🛠️ Step 1: Launch an EC2 Instance

1.  **Go to AWS EC2 Console** and click **Launch Instance**.
2.  **Name**: `cift-production`
3.  **OS Image**: Choose **Ubuntu Server 22.04 LTS** (x86 architecture).
4.  **Instance Type**:
    *   **Recommended**: `t3.xlarge` (4 vCPU, 16GB RAM) - Best for running all databases (QuestDB, ClickHouse, Postgres, Dragonfly).
    *   **Minimum**: `t3.large` (2 vCPU, 8GB RAM) - Might struggle with all services running.
5.  **Key Pair**: Create a new key pair (e.g., `cift-key`) and download the `.pem` file.
6.  **Network Settings**:
    *   Allow SSH traffic from **My IP** (for security).
    *   Allow HTTP (80) and HTTPS (443) from **Anywhere**.
    *   Allow Custom TCP `3000` (Frontend), `8000` (API), `9000` (QuestDB Console) if you want to access them directly.
7.  **Storage**: Increase the root volume to at least **50GB** (gp3) to accommodate Docker images and database data.
8.  **Launch Instance**.

---

## 🔑 Step 2: Configure GitHub Secrets

To allow GitHub Actions to deploy to your server, you need to add secrets to your repository.

1.  Go to your GitHub Repo -> **Settings** -> **Secrets and variables** -> **Actions**.
2.  Click **New repository secret** and add the following:

| Secret Name | Value |
| :--- | :--- |
| `EC2_HOST` | The **Public IPv4 address** of your EC2 instance (e.g., `54.123.45.67`). |
| `EC2_USERNAME` | `ubuntu` |
| `EC2_SSH_KEY` | The content of your `.pem` key file. Open it with a text editor and copy everything from `-----BEGIN RSA PRIVATE KEY-----` to `-----END RSA PRIVATE KEY-----`. |
| `POSTGRES_PASSWORD` | (Optional) A strong password for your production database. |

---

## ⚙️ Step 3: Server Setup

1.  **SSH into your server**:
    ```bash
    ssh -i path/to/cift-key.pem ubuntu@YOUR_EC2_IP
    ```

2.  **Copy the setup script**:
    You can copy the content of `scripts/setup-ec2.sh` to a file on the server, or just run these commands:

    ```bash
    # Create the script
    nano setup.sh
    # Paste the content of scripts/setup-ec2.sh here
    # Save and exit (Ctrl+O, Enter, Ctrl+X)

    # Make it executable and run
    chmod +x setup.sh
    ./setup.sh
    ```

    *This script installs Docker, Docker Compose, Git, and sets up a Swap file.*

3.  **Clone/Link your Repo**:
    The script creates a `~/cift-markets` directory. You should clone your repo there:
    ```bash
    rm -rf ~/cift-markets
    git clone https://github.com/YOUR_USERNAME/cift-markets.git ~/cift-markets
    ```

---

## 🚀 Step 4: First Deployment

1.  **Push your code** to the `main` branch on GitHub.
2.  Go to the **Actions** tab in your GitHub repository.
3.  You should see the **Deploy to Production** workflow running.
4.  Once it completes (green checkmark), your server will pull the new images and restart.

### 🔍 Verifying Deployment

Visit `http://YOUR_EC2_IP:3000` to see the frontend.
Visit `http://YOUR_EC2_IP:8000/docs` to see the API documentation.

---

## 📝 Important Notes

*   **Image Names**: The `docker-compose.yml` is configured to pull images from `ghcr.io/mesof/cift-markets/...`. If your GitHub username or repo name is different, please update the `image:` fields in `docker-compose.yml` and the `deploy.yml` workflow.
*   **Database Persistence**: Data is stored in Docker volumes (`questdb-data`, `postgres-data`, etc.). It will persist across deployments.
*   **Security**: For a real production setup, you should set up Nginx as a reverse proxy with SSL (Let's Encrypt) in front of your services, instead of exposing ports 3000/8000 directly.
