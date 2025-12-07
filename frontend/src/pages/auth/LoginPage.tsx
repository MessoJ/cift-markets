/**
 * Login Page
 * 
 * Modern, glassmorphic login interface with animations.
 */

import { createSignal, Show } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { Mail, Lock, AlertCircle, TrendingUp } from 'lucide-solid';
import { Logo } from '~/components/layout/Logo';
import { Input } from '~/components/ui/Input';
import { Button } from '~/components/ui/Button';
import { authStore } from '~/stores/auth.store';

export default function LoginPage() {
  const navigate = useNavigate();
  
  const [email, setEmail] = createSignal('');
  const [password, setPassword] = createSignal('');
  const [error, setError] = createSignal('');
  const [loading, setLoading] = createSignal(false);

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await authStore.login(email(), password());
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="min-h-screen bg-gray-950 flex items-center justify-center p-3 sm:p-4 md:p-6 relative overflow-hidden">
      {/* Animated Background */}
      <div class="absolute inset-0 overflow-hidden">
        <div class="absolute top-0 left-1/4 w-64 h-64 sm:w-96 sm:h-96 bg-primary-500/10 rounded-full blur-3xl animate-pulse" />
        <div class="absolute bottom-0 right-1/4 w-64 h-64 sm:w-96 sm:h-96 bg-success-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
      </div>

      {/* Left Side - Branding */}
      <div class="hidden lg:flex flex-1 items-center justify-center relative z-10 px-4">
        <div class="max-w-md space-y-6 lg:space-y-8 animate-fade-in">
          <Logo size="lg" showText />
          
          <div class="space-y-3 sm:space-y-4">
            <h1 class="text-2xl sm:text-3xl lg:text-4xl font-bold text-white">
              Institutional Trading Platform
            </h1>
            <p class="text-base lg:text-lg text-gray-400">
              Advanced algorithmic trading with real-time analytics and sub-10ms execution.
            </p>
          </div>

          <div class="space-y-3 sm:space-y-4">
            <div class="flex items-start gap-3 sm:gap-4 glass-bg p-3 sm:p-4 rounded-lg">
              <div class="w-10 h-10 rounded-lg bg-primary-500/20 flex items-center justify-center flex-shrink-0">
                <TrendingUp class="w-5 h-5 text-primary-400" />
              </div>
              <div>
                <h3 class="font-semibold text-white mb-1">Lightning Fast Execution</h3>
                <p class="text-sm text-gray-400">
                  Sub-10ms latency with Phase 5-7 tech stack
                </p>
              </div>
            </div>

            <div class="flex items-start gap-4 glass-bg p-4 rounded-lg">
              <div class="w-10 h-10 rounded-lg bg-success-500/20 flex items-center justify-center flex-shrink-0">
                <svg class="w-5 h-5 text-success-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h3 class="font-semibold text-white mb-1">Advanced Analytics</h3>
                <p class="text-sm text-gray-400">
                  Real-time P&L, risk metrics, and performance tracking
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div class="w-full max-w-md relative z-10">
        <div class="glass-bg rounded-2xl p-8 shadow-2xl animate-slide-up">
          {/* Mobile Logo */}
          <div class="lg:hidden mb-8 flex justify-center">
            <Logo size="lg" variant="full" theme="dark" animated />
          </div>

          <div class="mb-8">
            <h2 class="text-2xl font-bold text-white mb-2">
              Welcome back
            </h2>
            <p class="text-gray-400">
              Sign in to your account to continue
            </p>
          </div>

          <Show when={error()}>
            <div class="mb-6 p-4 bg-danger-900/20 border border-danger-700/50 rounded-lg flex items-start gap-3 animate-slide-down">
              <AlertCircle class="w-5 h-5 text-danger-400 flex-shrink-0 mt-0.5" />
              <div class="flex-1">
                <p class="text-sm text-danger-300">{error()}</p>
              </div>
            </div>
          </Show>

          <form onSubmit={handleSubmit} class="space-y-5">
            <Input
              type="email"
              label="Email"
              placeholder="your@email.com"
              value={email()}
              onInput={(e) => setEmail(e.currentTarget.value)}
              leftIcon={<Mail class="w-4 h-4" />}
              required
              fullWidth
              autocomplete="email"
            />

            <Input
              type="password"
              label="Password"
              placeholder="••••••••"
              value={password()}
              onInput={(e) => setPassword(e.currentTarget.value)}
              leftIcon={<Lock class="w-4 h-4" />}
              required
              fullWidth
              autocomplete="current-password"
            />

            <div class="flex items-center justify-between">
              <label class="flex items-center gap-2 cursor-pointer group">
                <input
                  type="checkbox"
                  class="w-4 h-4 rounded border-gray-700 bg-gray-800 text-primary-500 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-gray-950 transition-colors"
                />
                <span class="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">
                  Remember me
                </span>
              </label>

              <a
                href="/auth/forgot-password"
                class="text-sm text-primary-400 hover:text-primary-300 transition-colors"
              >
                Forgot password?
              </a>
            </div>

            <Button
              type="submit"
              variant="primary"
              fullWidth
              loading={loading()}
              size="lg"
              class="mt-6"
            >
              Sign in
            </Button>
          </form>

          <div class="mt-6 text-center">
            <p class="text-sm text-gray-400">
              Don't have an account?{' '}
              <a
                href="/auth/register"
                class="text-primary-400 hover:text-primary-300 font-medium transition-colors"
              >
                Sign up
              </a>
            </p>
          </div>

          {/* Demo Credentials Info */}
          <div class="mt-8 p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
            <p class="text-xs text-gray-400 mb-2">Demo Account:</p>
            <div class="space-y-1 text-xs font-mono">
              <p class="text-gray-300">Email: admin@ciftmarkets.com</p>
              <p class="text-gray-300">Password: admin</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div class="mt-6 text-center text-xs text-gray-500">
          <p>
            By signing in, you agree to our{' '}
            <a href="/terms" class="hover:text-gray-400 transition-colors">Terms</a>
            {' '}and{' '}
            <a href="/privacy" class="hover:text-gray-400 transition-colors">Privacy Policy</a>
          </p>
        </div>
      </div>
    </div>
  );
}
