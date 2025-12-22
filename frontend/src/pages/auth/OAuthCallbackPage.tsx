import { useNavigate, useSearchParams } from "@solidjs/router";
import { onMount } from "solid-js";
import { authStore } from "~/stores/auth.store";

export default function OAuthCallbackPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  onMount(async () => {
    const accessToken = searchParams.access_token;
    const refreshToken = searchParams.refresh_token;

    if (accessToken && refreshToken) {
      try {
        // Store tokens
        localStorage.setItem("cift_access_token", accessToken);
        localStorage.setItem("cift_refresh_token", refreshToken);

        // Initialize auth store (this will fetch user profile)
        await authStore.checkAuth();

        // Redirect to dashboard
        navigate("/dashboard");
      } catch (error) {
        console.error("Failed to complete OAuth login:", error);
        navigate("/auth/login?error=oauth_failed");
      }
    } else {
      console.error("Missing tokens in callback URL");
      navigate("/auth/login?error=missing_tokens");
    }
  });

  return (
    <div class="min-h-screen bg-black flex items-center justify-center text-white">
      <div class="flex flex-col items-center gap-4">
        <div class="w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full animate-spin"></div>
        <p class="text-gray-400 font-mono">Completing secure login...</p>
      </div>
    </div>
  );
}
