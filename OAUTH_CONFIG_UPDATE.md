# OAuth Configuration Update

I have updated the configuration to include the GitHub OAuth credentials and the Frontend URL for the Azure VM.

## Changes Applied

1.  **`cift/core/config.py`**: Added fields for OAuth credentials and URLs to the `Settings` class.
2.  **`docker-compose.yml`**: Added the following environment variables to the `api` service with the provided values as defaults:
    *   `GITHUB_CLIENT_ID`: `Ov23liKhbfcXw0MTCR97`
    *   `GITHUB_CLIENT_SECRET`: `f5377d213e5171b1833938b839fcdb17b563405d`
    *   `FRONTEND_URL`: `http://20.250.40.67:3000`
    *   `API_BASE_URL`: `http://20.250.40.67:8000`

## Next Steps

1.  **Push Changes:** Commit and push these changes to your GitHub repository.
    ```bash
    git add .
    git commit -m "Configure GitHub OAuth and Azure URLs"
    git push origin main
    ```
2.  **Deploy:** The GitHub Actions workflow should pick up the changes and deploy to the Azure VM.
3.  **Verify:** Once deployed, try logging in with GitHub on the live site.

**Note:** Hardcoding secrets in `docker-compose.yml` is generally not recommended for production security, but it will work for your current setup. For better security in the future, consider using GitHub Secrets and passing them as environment variables during the build/deploy process.
