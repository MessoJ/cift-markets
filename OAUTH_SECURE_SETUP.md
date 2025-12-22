# OAuth Configuration Guide

I have updated the code to **NOT** hardcode secrets. Instead, they will be injected securely during deployment.

## 1. Set GitHub Secrets
You must go to your GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions** and add the following secrets:

| Name | Value |
| :--- | :--- |
| `GITHUB_CLIENT_ID` | `Ov23liKhbfcXw0MTCR97` |
| `GITHUB_CLIENT_SECRET` | `f5377d213e5171b1833938b839fcdb17b563405d` |
| `FRONTEND_URL` | `http://20.250.40.67:3000` |
| `API_BASE_URL` | `http://20.250.40.67:8000` |

## 2. How it Works
1.  **`docker-compose.yml`**: Now uses variables like `${GITHUB_CLIENT_ID}` instead of hardcoded values.
2.  **`deploy.yml`**: I updated the deployment script to create a `.env` file on the Azure VM using the secrets you set in GitHub.
3.  **Deployment**: When you push code, GitHub Actions will SSH into the VM, create the `.env` file with your secrets, and then start the containers. The containers will read the secrets from that file.

## 3. Next Steps
1.  **Add the secrets** in GitHub Settings as shown above.
2.  **Commit and Push** the changes.
    ```bash
    git add .
    git commit -m "Configure secure OAuth deployment"
    git push origin main
    ```
