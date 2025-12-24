# GitHub Login Troubleshooting Guide

If you are still experiencing issues with GitHub login, please follow these steps to verify your configuration.

## 1. Check Server Environment Variables
The API needs the following environment variables to be set correctly in your `.env` file on the Azure server (`~/cift-markets/.env`).

```bash
# GitHub OAuth Credentials
GITHUB_CLIENT_ID=your_client_id_here
GITHUB_CLIENT_SECRET=your_client_secret_here

# URLs
FRONTEND_URL=http://20.250.40.67:3000
API_BASE_URL=http://20.250.40.67:8000
```

**Action:** SSH into your server and check the file:
```bash
ssh azureuser@20.250.40.67
cat ~/cift-markets/.env
```
Ensure `FRONTEND_URL` does **not** have a trailing slash.

## 2. Check GitHub OAuth App Settings
Go to your [GitHub Developer Settings](https://github.com/settings/developers) -> OAuth Apps -> Your App.

Verify the following:
-   **Homepage URL:** `http://20.250.40.67:3000`
-   **Authorization callback URL:** `http://20.250.40.67:8000/api/v1/auth/github/callback`

**Important:** The callback URL must match exactly what the API expects.

## 3. Check API Logs
I have added logging to the API startup to print the configured URLs. Restart the API and check the logs:

```bash
sudo docker restart cift-api
sudo docker logs cift-api --tail 50
```

Look for lines like:
```
INFO:     Frontend URL: http://20.250.40.67:3000
INFO:     API Base URL: http://20.250.40.67:8000
```

## 4. Common Issues
-   **Redirect Mismatch:** If the `redirect_uri` sent by the API doesn't match the one in GitHub settings, GitHub will reject the login.
-   **CORS:** If the frontend cannot talk to the API, the login flow might break. The API is configured to allow `http://20.250.40.67:3000`.
-   **Frontend Callback:** The frontend expects to receive tokens in the URL query parameters at `/auth/callback`.
