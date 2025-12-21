/**
 * Registration Page
 * 
 * Professional sign-up interface with real-time validation and security checks.
 */

import { createSignal, Show, createEffect } from 'solid-js';
import { useNavigate, A } from '@solidjs/router';
import { 
  Mail, 
  Lock, 
  User, 
  AlertCircle, 
  CheckCircle2, 
  ArrowRight, 
  ShieldCheck,
  Globe,
  Terminal
} from 'lucide-solid';
import { Logo } from '~/components/layout/Logo';
import { authStore } from '~/stores/auth.store';

export default function RegisterPage() {
  const navigate = useNavigate();
  
  // Form State
  const [fullName, setFullName] = createSignal('');
  const [username, setUsername] = createSignal('');
  const [email, setEmail] = createSignal('');
  const [password, setPassword] = createSignal('');
  const [confirmPassword, setConfirmPassword] = createSignal('');
  const [termsAccepted, setTermsAccepted] = createSignal(false);
  
  // UI State
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal('');
  const [passwordStrength, setPasswordStrength] = createSignal(0);

  // Password Strength Calculation
  createEffect(() => {
    const pwd = password();
    let score = 0;
    if (pwd.length > 8) score++;
    if (/[A-Z]/.test(pwd)) score++;
    if (/[0-9]/.test(pwd)) score++;
    if (/[^A-Za-z0-9]/.test(pwd)) score++;
    setPasswordStrength(score);
  });

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    setError('');

    if (password() !== confirmPassword()) {
      setError('Passwords do not match');
      return;
    }

    if (!termsAccepted()) {
      setError('You must accept the Terms of Service');
      return;
    }

    setLoading(true);

    try {
      await authStore.register(email(), username(), password(), fullName());
      // Redirect to onboarding flow instead of dashboard
      navigate('/onboarding');
    } catch (err: any) {
      setError(err.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="min-h-screen bg-black flex relative overflow-hidden font-sans text-white">
      {/* Background Grid */}
      <div class="absolute inset-0 bg-[url('/grid.png')] opacity-20 pointer-events-none"></div>
      
      {/* Left Side - Visuals */}
      <div class="hidden lg:flex flex-1 flex-col justify-between p-12 relative z-10 border-r border-terminal-800 bg-terminal-950/50 backdrop-blur-sm">
        <div>
          <Logo size="lg" showText />
          <div class="mt-12 space-y-6">
            <h1 class="text-5xl font-bold tracking-tight leading-tight">
              Join the <span class="text-accent-500">Future</span> of<br />
              Institutional Trading.
            </h1>
            <p class="text-xl text-gray-400 max-w-lg">
              Access Phase 5-7 tech stack, sub-10ms execution, and real-time global market analytics.
            </p>
          </div>
        </div>

        <div class="space-y-6">
          <div class="flex items-center gap-4 p-4 bg-terminal-900/50 border border-terminal-800 rounded-lg backdrop-blur-md">
            <div class="p-3 bg-accent-500/10 rounded-lg text-accent-400">
              <ShieldCheck size={24} />
            </div>
            <div>
              <h3 class="font-bold font-mono text-white">Bank-Grade Security</h3>
              <p class="text-sm text-gray-400">256-bit encryption & cold storage</p>
            </div>
          </div>
          
          <div class="flex items-center gap-4 p-4 bg-terminal-900/50 border border-terminal-800 rounded-lg backdrop-blur-md">
            <div class="p-3 bg-accent-500/10 rounded-lg text-accent-400">
              <Globe size={24} />
            </div>
            <div>
              <h3 class="font-bold font-mono text-white">Global Access</h3>
              <p class="text-sm text-gray-400">Trade across 50+ markets worldwide</p>
            </div>
          </div>
        </div>

        <div class="text-xs font-mono text-gray-500">
          © 2025 CIFT Markets. All systems operational.
        </div>
      </div>

      {/* Right Side - Form */}
      <div class="flex-1 flex items-center justify-center p-6 relative z-10">
        <div class="w-full max-w-md space-y-8">
          {/* Mobile Logo */}
          <div class="lg:hidden flex justify-center mb-8">
            <Logo size="lg" showText />
          </div>

          <div class="text-center lg:text-left">
            <h2 class="text-3xl font-bold tracking-tight">Create Account</h2>
            <p class="mt-2 text-gray-400">Enter your details to get started.</p>
          </div>

          <Show when={error()}>
            <div class="p-4 bg-danger-900/20 border border-danger-800 rounded-lg flex items-start gap-3 animate-in fade-in slide-in-from-top-2">
              <AlertCircle class="text-danger-500 shrink-0 mt-0.5" size={18} />
              <p class="text-sm text-danger-200">{error()}</p>
            </div>
          </Show>

          <form onSubmit={handleSubmit} class="space-y-5">
            <div class="grid grid-cols-2 gap-4">
              <div class="space-y-1.5">
                <label class="text-xs font-mono font-bold text-gray-400 uppercase">Full Name</label>
                <div class="relative">
                  <User class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                  <input
                    type="text"
                    value={fullName()}
                    onInput={(e) => setFullName(e.currentTarget.value)}
                    class="w-full bg-terminal-900 border border-terminal-800 rounded-lg py-2.5 pl-10 pr-4 text-white placeholder:text-gray-600 focus:border-accent-500 focus:ring-1 focus:ring-accent-500 outline-none transition-all font-mono text-sm"
                    placeholder="John Doe"
                  />
                </div>
              </div>
              <div class="space-y-1.5">
                <label class="text-xs font-mono font-bold text-gray-400 uppercase">Username</label>
                <div class="relative">
                  <Terminal class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                  <input
                    type="text"
                    value={username()}
                    onInput={(e) => setUsername(e.currentTarget.value)}
                    class="w-full bg-terminal-900 border border-terminal-800 rounded-lg py-2.5 pl-10 pr-4 text-white placeholder:text-gray-600 focus:border-accent-500 focus:ring-1 focus:ring-accent-500 outline-none transition-all font-mono text-sm"
                    placeholder="johndoe"
                    required
                  />
                </div>
              </div>
            </div>

            <div class="space-y-1.5">
              <label class="text-xs font-mono font-bold text-gray-400 uppercase">Email Address</label>
              <div class="relative">
                <Mail class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                <input
                  type="email"
                  value={email()}
                  onInput={(e) => setEmail(e.currentTarget.value)}
                  class="w-full bg-terminal-900 border border-terminal-800 rounded-lg py-2.5 pl-10 pr-4 text-white placeholder:text-gray-600 focus:border-accent-500 focus:ring-1 focus:ring-accent-500 outline-none transition-all font-mono text-sm"
                  placeholder="name@company.com"
                  required
                />
              </div>
            </div>

            <div class="space-y-1.5">
              <label class="text-xs font-mono font-bold text-gray-400 uppercase">Password</label>
              <div class="relative">
                <Lock class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                <input
                  type="password"
                  value={password()}
                  onInput={(e) => setPassword(e.currentTarget.value)}
                  class="w-full bg-terminal-900 border border-terminal-800 rounded-lg py-2.5 pl-10 pr-4 text-white placeholder:text-gray-600 focus:border-accent-500 focus:ring-1 focus:ring-accent-500 outline-none transition-all font-mono text-sm"
                  placeholder="••••••••"
                  required
                />
              </div>
              {/* Password Strength Meter */}
              <div class="flex gap-1 h-1 mt-2">
                <div class={`flex-1 rounded-full transition-colors ${passwordStrength() >= 1 ? 'bg-danger-500' : 'bg-terminal-800'}`} />
                <div class={`flex-1 rounded-full transition-colors ${passwordStrength() >= 2 ? 'bg-warning-500' : 'bg-terminal-800'}`} />
                <div class={`flex-1 rounded-full transition-colors ${passwordStrength() >= 3 ? 'bg-success-500' : 'bg-terminal-800'}`} />
                <div class={`flex-1 rounded-full transition-colors ${passwordStrength() >= 4 ? 'bg-accent-500' : 'bg-terminal-800'}`} />
              </div>
            </div>

            <div class="space-y-1.5">
              <label class="text-xs font-mono font-bold text-gray-400 uppercase">Confirm Password</label>
              <div class="relative">
                <Lock class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                <input
                  type="password"
                  value={confirmPassword()}
                  onInput={(e) => setConfirmPassword(e.currentTarget.value)}
                  class="w-full bg-terminal-900 border border-terminal-800 rounded-lg py-2.5 pl-10 pr-4 text-white placeholder:text-gray-600 focus:border-accent-500 focus:ring-1 focus:ring-accent-500 outline-none transition-all font-mono text-sm"
                  placeholder="••••••••"
                  required
                />
              </div>
            </div>

            <div class="flex items-start gap-3 pt-2">
              <div class="flex items-center h-5">
                <input
                  id="terms"
                  type="checkbox"
                  checked={termsAccepted()}
                  onChange={(e) => setTermsAccepted(e.currentTarget.checked)}
                  class="w-4 h-4 rounded border-terminal-700 bg-terminal-900 text-accent-500 focus:ring-accent-500 focus:ring-offset-terminal-950"
                />
              </div>
              <label for="terms" class="text-sm text-gray-400">
                I agree to the <a href="#" class="text-accent-400 hover:text-accent-300">Terms of Service</a> and <a href="#" class="text-accent-400 hover:text-accent-300">Privacy Policy</a>.
              </label>
            </div>

            <button
              type="submit"
              disabled={loading()}
              class="w-full bg-accent-600 hover:bg-accent-500 disabled:bg-terminal-800 disabled:text-gray-500 text-white font-bold py-3 rounded-lg transition-all flex items-center justify-center gap-2 group"
            >
              <Show when={loading()} fallback={
                <>
                  Create Account
                  <ArrowRight size={18} class="group-hover:translate-x-1 transition-transform" />
                </>
              }>
                <div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Processing...
              </Show>
            </button>
          </form>

          <div class="relative">
            <div class="absolute inset-0 flex items-center">
              <div class="w-full border-t border-terminal-800"></div>
            </div>
            <div class="relative flex justify-center text-xs uppercase">
              <span class="bg-black px-2 text-gray-500 font-mono">Or continue with</span>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <button class="flex items-center justify-center gap-2 bg-terminal-900 border border-terminal-800 hover:bg-terminal-800 text-white py-2.5 rounded-lg transition-colors text-sm font-medium">
              <svg class="w-5 h-5" viewBox="0 0 24 24">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              Google
            </button>
            <button class="flex items-center justify-center gap-2 bg-terminal-900 border border-terminal-800 hover:bg-terminal-800 text-white py-2.5 rounded-lg transition-colors text-sm font-medium">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M13.0729 1.04175C13.2083 1.04175 13.3333 1.09383 13.4271 1.18758C13.5208 1.28133 13.5729 1.40633 13.5729 1.54175V1.54175C13.5729 1.67717 13.5208 1.80217 13.4271 1.89592C13.3333 1.98967 13.2083 2.04175 13.0729 2.04175H13.0729C12.9375 2.04175 12.8125 1.98967 12.7188 1.89592C12.625 1.80217 12.5729 1.67717 12.5729 1.54175V1.54175C12.5729 1.40633 12.625 1.28133 12.7188 1.18758C12.8125 1.09383 12.9375 1.04175 13.0729 1.04175ZM13.0729 1.04175C13.2083 1.04175 13.3333 1.09383 13.4271 1.18758C13.5208 1.28133 13.5729 1.40633 13.5729 1.54175V1.54175C13.5729 1.67717 13.5208 1.80217 13.4271 1.89592C13.3333 1.98967 13.2083 2.04175 13.0729 2.04175H13.0729C12.9375 2.04175 12.8125 1.98967 12.7188 1.89592C12.625 1.80217 12.5729 1.67717 12.5729 1.54175V1.54175C12.5729 1.40633 12.625 1.28133 12.7188 1.18758C12.8125 1.09383 12.9375 1.04175 13.0729 1.04175Z" />
                <path d="M17.05 20.28c-.98.95-2.05 1.96-3.5 1.96-1.48 0-2.04-.9-3.85-.9-1.8 0-2.38.9-3.84.9-1.43 0-2.54-1.02-3.59-2.08-2.1-2.12-3.56-5.82-1.48-9.45 1.05-1.82 2.93-2.98 4.98-2.98 1.55 0 2.94 1.05 3.86 1.05.92 0 2.66-1.05 4.48-1.05 1.53 0 2.9.62 3.85 1.62-3.38 1.68-2.82 6.12.58 7.53-.75 1.49-1.8 3.38-3.5 5.3zM14.9 5.15c.82-1.02 1.38-2.45 1.22-3.88-1.18.05-2.6.78-3.45 1.8-.78.92-1.45 2.42-1.28 3.82 1.32.1 2.68-.72 3.5-1.74z" />
              </svg>
              Apple
            </button>
          </div>

          <div class="text-center">
            <p class="text-sm text-gray-400">
              Already have an account?{' '}
              <A href="/auth/login" class="text-accent-400 hover:text-accent-300 font-bold transition-colors">
                Sign in
              </A>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}