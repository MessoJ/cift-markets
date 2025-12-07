# 🚀 Deploying CIFT Markets to Azure

This guide explains how to host the CIFT Markets platform on Microsoft Azure using Docker Compose and GitHub Actions for automated deployments.

## 📋 Prerequisites

1.  **Azure Account**: You need an active Azure subscription.
2.  **Azure CLI**: Install the Azure CLI tool.
3.  **GitHub Repository**: This code must be pushed to a GitHub repository.

---

## 🛠️ Step 1: Provision Azure Infrastructure

We have provided a PowerShell script to automate the server creation.

1.  **Login to Azure**:
    ```powershell
    az login
    ```

2.  **Run the Provisioning Script**:
    ```powershell
    .\scripts\provision-azure.ps1
    ```
    *   This will create a Resource Group `cift-resources`.
    *   It will launch an Ubuntu VM `cift-production`.
    *   It will open ports 80, 443, 3000, 8000, and 9000.
    *   **Note the Public IP** output at the end.

---

## 🔑 Step 2: Configure GitHub Secrets

To allow GitHub Actions to deploy to your server, you need to add secrets to your repository.

1.  Go to your GitHub Repo -> **Settings** -> **Secrets and variables** -> **Actions**.
2.  Click **New repository secret** and add the following:

| Secret Name | Value |
| :--- | :--- |
| `AZURE_VM_IP` | The **Public IP** of your Azure VM (output from the script). |
| `AZURE_VM_USERNAME` | `azureuser` (default) |
| `AZURE_VM_SSH_KEY` | The content of your private SSH key. Usually found at `~/.ssh/id_rsa` on the machine where you ran the script. Copy the full content. |
| `POSTGRES_PASSWORD` | (Optional) A strong password for your production database. |

---

## ⚙️ Step 3: Server Setup

1.  **SSH into your server**:
    ```bash
    ssh azureuser@YOUR_VM_IP
    ```

2.  **Copy the setup script**:
    You can copy the content of `scripts/setup-ec2.sh` (it works for Azure too, as it's just Ubuntu setup) to a file on the server:

    ```bash
    # Create the script
    nano setup.sh
    # Paste the content of scripts/setup-ec2.sh here
    # Save and exit (Ctrl+O, Enter, Ctrl+X)

    # Make it executable and run
    chmod +x setup.sh
    ./setup.sh
    ```

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

Visit `http://YOUR_VM_IP:3000` to see the frontend.
Visit `http://YOUR_VM_IP:8000/docs` to see the API documentation.
