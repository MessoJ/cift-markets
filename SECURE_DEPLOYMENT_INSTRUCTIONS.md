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
| `GH_CLIENT_ID` | `Ov23liKhbfcXw0MTCR97` |
| `GH_CLIENT_SECRET` | `f5377d213e5171b1833938b839fcdb17b563405d` |
| `FRONTEND_URL` | `http://20.250.40.67:3000` |
| `API_BASE_URL` | `http://20.250.40.67:8000` |

### Application Secrets (From your .env)
| Name | Value |
| :--- | :--- |
| `POSTGRES_PASSWORD` | `changeme123` |
| `JWT_SECRET_KEY` | `M6xuob6TCLasgTUuQTsTpJEdNS_icxLoKdJSzw0hJOJFalWzhEw3OxA-y0CWUv0d` |
| `SECRET_KEY` | `8erB62ICTl0RmSHhQWvcFsaVnzGrLNhb6iC0HUQ55wJn1E119uJHG-iT2ganaI_A` |
| `NEWSAPI_KEY` | `d888b7199f80455abfd9b27e47cec5bb` |
| `FINNHUB_API_KEY` | `d4ojf7pr01qtc1p01m60d4ojf7pr01qtc1p01m6g` |
| `ALPHAVANTAGE_API_KEY` | `4V9I2E4N97HN7ZFV` |
| `ALPACA_API_KEY` | `PKZJ7H3NMIIXLERTEILU4MLRMZ` |
| `ALPACA_SECRET_KEY` | `3fcyzA6Q46sQt67wMfgXKovgF28LLc8fUkg8fRzDR6Tz` |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` |
| `POLYGON_API_KEY` | `lbAovcp5kn4vqPJ3H_1qcJrzYiwmUhye` |

## 3. Deploy
Once you have added the secrets above, simply push the changes to deploy:

```bash
git add .
git commit -m "Secure OAuth configuration via GitHub Secrets"
git push origin main
```

This ensures your secrets are never stored in the code history and are only present on the server where they are needed.
