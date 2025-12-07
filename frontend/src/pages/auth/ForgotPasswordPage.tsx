/**
 * Forgot Password Page
 * 
 * Interface for password recovery.
 */

import { createSignal, Show } from 'solid-js';
import { A } from '@solidjs/router';
import { Mail, ArrowRight, CheckCircle2, AlertCircle } from 'lucide-solid';
import { Logo } from '~/components/layout/Logo';

export default function ForgotPasswordPage() {
  const [email, setEmail] = createSignal('');
  const [submitted, setSubmitted] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal('');

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setSubmitted(true);
    }, 1500);
  };

  return (
    <div class="min-h-screen bg-black flex items-center justify-center p-6 relative overflow-hidden font-sans text-white">
      {/* Background Grid */}
      <div class="absolute inset-0 bg-[url('/grid.png')] opacity-20 pointer-events-none"></div>

      <div class="w-full max-w-md relative z-10">
        <div class="text-center mb-8">
          <div class="flex justify-center mb-6">
            <Logo size="lg" showText />
          </div>
          <h2 class="text-2xl font-bold tracking-tight">Reset Password</h2>
          <p class="mt-2 text-gray-400">Enter your email to receive recovery instructions.</p>
        </div>

        <Show when={!submitted()} fallback={
          <div class="bg-terminal-900/50 border border-terminal-800 rounded-xl p-8 text-center backdrop-blur-sm animate-in fade-in zoom-in-95">
            <div class="w-16 h-16 bg-success-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 class="text-success-500" size={32} />
            </div>
            <h3 class="text-xl font-bold mb-2">Check your email</h3>
            <p class="text-gray-400 mb-6">
              We've sent password reset instructions to <span class="text-white font-mono">{email()}</span>
            </p>
            <A href="/auth/login" class="inline-flex items-center justify-center w-full bg-terminal-800 hover:bg-terminal-700 text-white font-bold py-3 rounded-lg transition-colors">
              Return to Login
            </A>
          </div>
        }>
          <div class="bg-terminal-900/50 border border-terminal-800 rounded-xl p-8 backdrop-blur-sm">
            <form onSubmit={handleSubmit} class="space-y-6">
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

              <button
                type="submit"
                disabled={loading()}
                class="w-full bg-accent-600 hover:bg-accent-500 disabled:bg-terminal-800 disabled:text-gray-500 text-white font-bold py-3 rounded-lg transition-all flex items-center justify-center gap-2 group"
              >
                <Show when={loading()} fallback={
                  <>
                    Send Reset Link
                    <ArrowRight size={18} class="group-hover:translate-x-1 transition-transform" />
                  </>
                }>
                  <div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Processing...
                </Show>
              </button>
            </form>

            <div class="mt-6 text-center">
              <A href="/auth/login" class="text-sm text-gray-400 hover:text-white transition-colors flex items-center justify-center gap-2">
                <ArrowRight class="rotate-180" size={14} />
                Back to Login
              </A>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}