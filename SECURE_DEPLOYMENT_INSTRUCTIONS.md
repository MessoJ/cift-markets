# Secure OAuth Deployment Guide

I have updated the deployment configuration to **avoid hardcoding secrets**. Instead, the secrets will be securely injected into the Azure VM during the deployment process using GitHub Actions.

## 1. Changes Made
- **`docker-compose.yml`**: Removed the hardcoded default values. It now expects environment variables to be present.
- **`.github/workflows/deploy.yml`**: Updated the deployment script to:
  1.  Read secrets from your GitHub Repository Secrets.
  2.  Pass them securely to the Azure VM via SSH.
  3.  Create a `.env` file on the VM containing these secrets before starting the containers.

## 2. Action Required: Set GitHub Secrets
You must now add the following secrets to your GitHub repository for the deployment to work.
**Note:** Secret names starting with `GITHUB_` are reserved, so we use `GH_` instead.

1.  Go to your GitHub Repository: **Settings** -> **Secrets and variables** -> **Actions**.
2.  Click **New repository secret**.
3.  Add the following secrets:

| Name | Value |
| :--- | :--- |
| `GH_CLIENT_ID` | `your_github_oauth_client_id` |
| `GH_CLIENT_SECRET` | `your_github_oauth_client_secret` |
| `FRONTEND_URL` | `http://20.250.40.67:3000` |
| `API_BASE_URL` | `http://20.250.40.67:8000` |

### Application Secrets (From your .env)
| Name | Value |
| :--- | :--- |
| `POSTGRES_PASSWORD` | `your_postgres_password` |
| `JWT_SECRET_KEY` | `your_jwt_secret_min_32_chars` |
| `SECRET_KEY` | `your_app_secret_min_32_chars` |
| `NEWSAPI_KEY` | `your_newsapi_key` |
| `FINNHUB_API_KEY` | `your_finnhub_key` |
| `ALPHAVANTAGE_API_KEY` | `your_alphavantage_key` |
| `ALPACA_API_KEY` | `your_alpaca_key` |
| `ALPACA_SECRET_KEY` | `your_alpaca_secret` |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` |
| `POLYGON_API_KEY` | `your_polygon_key` |

## 3. Deploy
Once you have added the secrets above, simply push the changes to deploy:

```bash
git add .
git commit -m "Secure OAuth configuration via GitHub Secrets"
git push origin main
```

This ensures your secrets are provided via GitHub Secrets and are only present on the server where they are needed. Do not paste real secret values into documentation.
