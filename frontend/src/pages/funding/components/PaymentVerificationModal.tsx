import { createSignal, Show, onMount, onCleanup } from 'solid-js';
import { X, CheckCircle, AlertCircle, Loader, Clock, ExternalLink, Smartphone } from 'lucide-solid';
import { apiClient, PaymentMethod } from '../../../lib/api/client';

interface PaymentVerificationModalProps {
  paymentMethod: PaymentMethod;
  onSuccess: () => void;
  onClose: () => void;
}

type VerificationState = {
  status: string;
  verification_type?: string;
  message: string;
  requires_action: boolean;
  action_type?: string;
  oauth_url?: string;
  expires_at?: string;
  attempt_count?: number;
  remaining_attempts?: number;
  error?: string;
};

export function PaymentVerificationModal(props: PaymentVerificationModalProps) {
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [verificationState, setVerificationState] = createSignal<VerificationState | null>(null);
  
  // Micro-deposit amounts
  const [amount1, setAmount1] = createSignal('');
  const [amount2, setAmount2] = createSignal('');
  
  // Verification code for other types
  const [verificationCode, setVerificationCode] = createSignal('');
  
  // Polling interval
  let pollInterval: number | null = null;

  onMount(async () => {
    await checkVerificationStatus();
    
    // Poll status every 3 seconds for real-time updates
    pollInterval = window.setInterval(() => {
      checkVerificationStatus();
    }, 3000);
  });

  onCleanup(() => {
    if (pollInterval) {
      clearInterval(pollInterval);
    }
  });

  const checkVerificationStatus = async () => {
    try {
      const status = await apiClient.getPaymentVerificationStatus(props.paymentMethod.id);
      
      setVerificationState({
        status: status.status,
        verification_type: status.verification_type,
        message: getStatusMessage(status),
        requires_action: needsUserAction(status.status),
        attempt_count: status.attempt_count,
        expires_at: status.expires_at,
        error: status.error,
      });

      // If verified, show success and close after delay
      if (status.is_verified) {
        setTimeout(() => {
          props.onSuccess();
          props.onClose();
        }, 2000);
      }
    } catch (err: any) {
      console.error('Failed to check verification status:', err);
    }
  };

  const getStatusMessage = (status: any): string => {
    if (status.is_verified) {
      return 'Payment method verified successfully!';
    }

    switch (status.verification_type) {
      case 'micro_deposit':
        return 'We\'ve sent two small deposits to your bank account. This may take 1-3 business days. Once received, enter the amounts below to verify.';
      case 'stk_push':
        return `A verification request has been sent to ${props.paymentMethod.mpesa_phone}. Please check your phone and enter your M-Pesa PIN to authorize.`;
      case 'oauth':
        if (props.paymentMethod.type === 'paypal') {
          return 'Click the button below to authorize CIFT Markets to connect to your PayPal account.';
        } else if (props.paymentMethod.type === 'cashapp') {
          return 'Click the button below to authorize CIFT Markets to connect to your Cash App account.';
        }
        return 'Authorization required to complete verification.';
      case 'instant':
        return 'Verifying payment method...';
      default:
        if (status.status === 'verification_failed') {
          return status.error || 'Verification failed. Please try again or contact support.';
        }
        return 'Verification in progress...';
    }
  };

  const needsUserAction = (status: string): boolean => {
    return status === 'awaiting_confirmation' || status === 'verification_initiated';
  };

  const handleMicroDepositSubmit = async () => {
    const amt1 = parseFloat(amount1());
    const amt2 = parseFloat(amount2());

    if (!amt1 || !amt2 || amt1 <= 0 || amt2 <= 0) {
      setError('Please enter valid amounts (e.g., 0.32)');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.completePaymentVerification(props.paymentMethod.id, {
        amount1: amt1,
        amount2: amt2,
      });

      if (result.status === 'verified') {
        setVerificationState({
          status: 'verified',
          verification_type: 'micro_deposit',
          message: 'Payment method verified successfully!',
          requires_action: false,
        });
        setTimeout(() => {
          props.onSuccess();
          props.onClose();
        }, 2000);
      } else {
        setError(result.message || 'Verification failed. Please check the amounts and try again.');
        if (result.remaining_attempts !== undefined) {
          setError(`${result.message} ${result.remaining_attempts} attempts remaining.`);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to verify amounts');
    } finally {
      setLoading(false);
    }
  };

  const handleStkPushConfirm = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.completePaymentVerification(props.paymentMethod.id, {
        confirmed: true,
      });

      if (result.status === 'verified') {
        setVerificationState({
          status: 'verified',
          verification_type: 'stk_push',
          message: 'M-Pesa account verified successfully!',
          requires_action: false,
        });
        setTimeout(() => {
          props.onSuccess();
          props.onClose();
        }, 2000);
      } else {
        setError(result.message || 'Verification failed.');
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to complete verification');
    } finally {
      setLoading(false);
    }
  };

  const handleOAuthAuthorize = () => {
    const state = verificationState();
    if (state?.oauth_url) {
      window.open(state.oauth_url, '_blank', 'width=600,height=700');
      // Continue polling to detect when OAuth completes
    }
  };

  const getVerificationIcon = () => {
    const state = verificationState();
    if (!state) return <Loader class="animate-spin" size={24} />;

    if (state.status === 'verified') {
      return <CheckCircle size={24} class="text-success-500" />;
    }

    if (state.status === 'verification_failed') {
      return <AlertCircle size={24} class="text-danger-500" />;
    }

    switch (state.verification_type) {
      case 'micro_deposit':
        return <Clock size={24} class="text-warning-500" />;
      case 'stk_push':
        return <Smartphone size={24} class="text-success-500" />;
      case 'oauth':
        return <ExternalLink size={24} class="text-info-500" />;
      default:
        return <Loader class="animate-spin" size={24} />;
    }
  };

  return (
    <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-2 sm:p-4">
      <div class="bg-terminal-900 border border-terminal-750 rounded-lg max-w-md w-full p-4 sm:p-6">
        {/* Header */}
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-base sm:text-lg font-bold text-white">Verify Payment Method</h2>
          <button
            onClick={props.onClose}
            class="text-gray-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Payment Method Info */}
        <div class="mb-4 p-2 sm:p-3 bg-terminal-850 border border-terminal-750 rounded">
          <div class="text-xs sm:text-sm text-gray-400 mb-1">
            {props.paymentMethod.type === 'bank_account' && 'Bank Account'}
            {(props.paymentMethod.type === 'debit_card' || props.paymentMethod.type === 'credit_card') && 'Card'}
            {props.paymentMethod.type === 'paypal' && 'PayPal'}
            {props.paymentMethod.type === 'cashapp' && 'Cash App'}
            {props.paymentMethod.type === 'mpesa' && 'M-Pesa'}
            {props.paymentMethod.type === 'crypto_wallet' && 'Crypto Wallet'}
          </div>
          <div class="text-white font-semibold">
            {props.paymentMethod.name || `••••${props.paymentMethod.last_four}`}
          </div>
        </div>

        {/* Status Icon & Message */}
        <div class="mb-6">
          <div class="flex justify-center mb-4">
            {getVerificationIcon()}
          </div>
          <Show when={verificationState()}>
            <p class="text-sm text-gray-300 text-center">
              {verificationState()!.message}
            </p>
            <Show when={verificationState()!.expires_at}>
              <p class="text-xs text-gray-500 text-center mt-2">
                Expires: {new Date(verificationState()!.expires_at!).toLocaleString()}
              </p>
            </Show>
          </Show>
        </div>

        {/* Verification Forms */}
        <Show when={verificationState()?.requires_action}>
          <div class="space-y-4">
            {/* Micro-Deposit Form */}
            <Show when={verificationState()?.verification_type === 'micro_deposit'}>
              <div class="space-y-3">
                <div>
                  <label class="block text-xs text-gray-400 mb-1">First Deposit Amount</label>
                  <div class="relative">
                    <span class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                    <input
                      type="number"
                      step="0.01"
                      value={amount1()}
                      onInput={(e) => setAmount1(e.currentTarget.value)}
                      placeholder="0.00"
                      class="w-full pl-7 pr-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
                    />
                  </div>
                </div>
                <div>
                  <label class="block text-xs text-gray-400 mb-1">Second Deposit Amount</label>
                  <div class="relative">
                    <span class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                    <input
                      type="number"
                      step="0.01"
                      value={amount2()}
                      onInput={(e) => setAmount2(e.currentTarget.value)}
                      placeholder="0.00"
                      class="w-full pl-7 pr-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
                    />
                  </div>
                </div>
                <button
                  onClick={handleMicroDepositSubmit}
                  disabled={loading() || !amount1() || !amount2()}
                  class="w-full bg-primary-500 hover:bg-primary-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-2 rounded transition-colors"
                >
                  {loading() ? 'Verifying...' : 'Verify Amounts'}
                </button>
              </div>
            </Show>

            {/* STK Push Confirmation */}
            <Show when={verificationState()?.verification_type === 'stk_push'}>
              <div class="space-y-3">
                <div class="p-3 bg-success-500/10 border border-success-500/20 rounded">
                  <p class="text-xs text-gray-300 text-center">
                    Check your phone for the M-Pesa prompt. After entering your PIN, click the button below.
                  </p>
                </div>
                <button
                  onClick={handleStkPushConfirm}
                  disabled={loading()}
                  class="w-full bg-success-500 hover:bg-success-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-2 rounded transition-colors"
                >
                  {loading() ? 'Confirming...' : 'I\'ve Authorized on My Phone'}
                </button>
              </div>
            </Show>

            {/* OAuth Authorization */}
            <Show when={verificationState()?.verification_type === 'oauth'}>
              <div class="space-y-3">
                <button
                  onClick={handleOAuthAuthorize}
                  class="w-full bg-info-500 hover:bg-info-600 text-white font-semibold py-2 rounded transition-colors flex items-center justify-center gap-2"
                >
                  <ExternalLink size={16} />
                  Authorize {props.paymentMethod.type === 'paypal' ? 'PayPal' : 'Cash App'}
                </button>
                <p class="text-xs text-gray-500 text-center">
                  A new window will open for authorization. This page will update automatically once complete.
                </p>
              </div>
            </Show>
          </div>
        </Show>

        {/* Error Message */}
        <Show when={error()}>
          <div class="mt-4 p-3 bg-danger-500/10 border border-danger-500/20 rounded">
            <div class="flex items-start gap-2">
              <AlertCircle size={16} class="text-danger-500 mt-0.5 flex-shrink-0" />
              <p class="text-xs text-danger-500">{error()}</p>
            </div>
          </div>
        </Show>

        {/* Close Button for Verified/Failed States */}
        <Show when={!verificationState()?.requires_action && verificationState()?.status !== 'verification_initiated'}>
          <button
            onClick={props.onClose}
            class="w-full mt-4 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-white font-semibold py-2 rounded transition-colors"
          >
            Close
          </button>
        </Show>
      </div>
    </div>
  );
}
