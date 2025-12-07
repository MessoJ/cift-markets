import { Component, Show, createSignal, onMount } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import { CheckCircle, XCircle, AlertCircle, ArrowLeft, Shield } from 'lucide-solid';

interface VerificationData {
  valid: boolean;
  transaction?: {
    id: string;
    type: string;
    status: string;
    amount: number;
    fee: number;
    created_at: string;
    payment_method_type?: string;
    payment_method_last4?: string;
  };
  message?: string;
}

export const VerifyTransactionPage: Component = () => {
  const params = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = createSignal(true);
  const [data, setData] = createSignal<VerificationData | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  onMount(async () => {
    const transactionId = params.id;
    
    if (!transactionId) {
      setError('No transaction ID provided');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/api/v1/verify/${transactionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to verify transaction');
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  });

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatAmount = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  return (
    <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div class="bg-white border-b border-slate-200 shadow-sm">
        <div class="max-w-4xl mx-auto px-4 py-6">
          <div class="flex items-center gap-3">
            <Shield class="w-8 h-8 text-blue-500" />
            <div>
              <h1 class="text-2xl font-bold text-slate-900">
                <span class="text-blue-500 font-bold">CIFT</span>{' '}
                <span class="text-slate-900">MARKETS</span>
              </h1>
              <p class="text-sm text-slate-600">Transaction Verification</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div class="max-w-4xl mx-auto px-4 py-12">
        <Show when={loading()}>
          <div class="bg-white rounded-xl shadow-lg p-12 text-center">
            <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
            <p class="mt-4 text-slate-600">Verifying transaction...</p>
          </div>
        </Show>

        <Show when={!loading() && error()}>
          <div class="bg-white rounded-xl shadow-lg p-8">
            <div class="flex items-center gap-4 mb-6">
              <XCircle class="w-12 h-12 text-red-500" />
              <div>
                <h2 class="text-2xl font-bold text-slate-900">Verification Failed</h2>
                <p class="text-slate-600">Unable to verify this transaction</p>
              </div>
            </div>
            <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <p class="text-red-800">{error()}</p>
            </div>
            <button
              onClick={() => navigate('/')}
              class="flex items-center gap-2 text-blue-500 hover:text-blue-600 font-medium"
            >
              <ArrowLeft class="w-4 h-4" />
              Return to Home
            </button>
          </div>
        </Show>

        <Show when={!loading() && data() && !data()!.valid}>
          <div class="bg-white rounded-xl shadow-lg p-8">
            <div class="flex items-center gap-4 mb-6">
              <AlertCircle class="w-12 h-12 text-amber-500" />
              <div>
                <h2 class="text-2xl font-bold text-slate-900">Transaction Not Found</h2>
                <p class="text-slate-600">This transaction could not be verified</p>
              </div>
            </div>
            <div class="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
              <p class="text-amber-800">
                {data()!.message || 'The transaction ID provided does not match our records.'}
              </p>
            </div>
            <p class="text-sm text-slate-600 mb-6">
              If you believe this is an error, please contact our support team at{' '}
              <a href="mailto:support@ciftmarkets.com" class="text-blue-500 hover:underline">
                support@ciftmarkets.com
              </a>{' '}
              or call{' '}
              <a href="tel:+16469782187" class="text-blue-500 hover:underline">
                +1 (646) 978-2187
              </a>
              .
            </p>
            <button
              onClick={() => navigate('/')}
              class="flex items-center gap-2 text-blue-500 hover:text-blue-600 font-medium"
            >
              <ArrowLeft class="w-4 h-4" />
              Return to Home
            </button>
          </div>
        </Show>

        <Show when={!loading() && data() && data()!.valid && data()!.transaction}>
          {(txn) => {
            const transaction = data()!.transaction!;
            const total = transaction.amount + transaction.fee;
            
            return (
              <div class="bg-white rounded-xl shadow-lg overflow-hidden">
                {/* Success Header */}
                <div class="bg-gradient-to-r from-green-500 to-emerald-500 p-8 text-white">
                  <div class="flex items-center gap-4 mb-4">
                    <CheckCircle class="w-16 h-16" />
                    <div>
                      <h2 class="text-3xl font-bold">Verified Transaction</h2>
                      <p class="text-green-100">This receipt is legitimate and issued by CIFT Markets</p>
                    </div>
                  </div>
                </div>

                {/* Transaction Details */}
                <div class="p-8">
                  <div class="grid md:grid-cols-2 gap-8 mb-8">
                    {/* Left Column */}
                    <div class="space-y-6">
                      <div>
                        <h3 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
                          Transaction Details
                        </h3>
                        <div class="space-y-3">
                          <div class="flex justify-between py-2 border-b border-slate-100">
                            <span class="text-slate-600">Transaction ID</span>
                            <span class="font-mono text-sm text-slate-900">
                              {transaction.id.slice(0, 8)}...{transaction.id.slice(-4)}
                            </span>
                          </div>
                          <div class="flex justify-between py-2 border-b border-slate-100">
                            <span class="text-slate-600">Date & Time</span>
                            <span class="font-medium text-slate-900">
                              {formatDate(transaction.created_at)}
                            </span>
                          </div>
                          <div class="flex justify-between py-2 border-b border-slate-100">
                            <span class="text-slate-600">Type</span>
                            <span class={`font-semibold ${
                              transaction.type === 'DEPOSIT' ? 'text-green-600' : 'text-blue-600'
                            }`}>
                              {transaction.type}
                            </span>
                          </div>
                          <div class="flex justify-between py-2 border-b border-slate-100">
                            <span class="text-slate-600">Status</span>
                            <span class={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold ${
                              transaction.status === 'COMPLETED' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-slate-100 text-slate-800'
                            }`}>
                              {transaction.status === 'COMPLETED' && <CheckCircle class="w-3 h-3" />}
                              {transaction.status}
                            </span>
                          </div>
                        </div>
                      </div>

                      <Show when={transaction.payment_method_type}>
                        <div>
                          <h3 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
                            Payment Method
                          </h3>
                          <div class="bg-slate-50 rounded-lg p-4">
                            <p class="text-slate-900 font-medium">
                              {transaction.payment_method_type?.replace('_', ' ')} 
                              <Show when={transaction.payment_method_last4}>
                                {' '}••••{transaction.payment_method_last4}
                              </Show>
                            </p>
                          </div>
                        </div>
                      </Show>
                    </div>

                    {/* Right Column - Amount Summary */}
                    <div>
                      <h3 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
                        Amount Summary
                      </h3>
                      <div class="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100">
                        <div class="space-y-3 mb-4">
                          <div class="flex justify-between text-slate-600">
                            <span>Subtotal</span>
                            <span class="font-medium">{formatAmount(transaction.amount)}</span>
                          </div>
                          <div class="flex justify-between text-slate-600">
                            <span>Processing Fee</span>
                            <span class="font-medium">{formatAmount(transaction.fee)}</span>
                          </div>
                        </div>
                        <div class="border-t-2 border-blue-200 pt-4">
                          <div class="flex justify-between items-center">
                            <span class="text-lg font-semibold text-slate-900">Total</span>
                            <span class="text-2xl font-bold text-blue-600">
                              {formatAmount(total)}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div class="mt-6 bg-green-50 border border-green-200 rounded-lg p-4">
                        <div class="flex items-start gap-3">
                          <Shield class="w-5 h-5 text-green-600 mt-0.5" />
                          <div>
                            <p class="text-sm font-semibold text-green-900 mb-1">
                              Verified & Secure
                            </p>
                            <p class="text-xs text-green-700">
                              This transaction has been verified against CIFT Markets' official records. 
                              All transaction data is authentic and accurate.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Full Transaction ID */}
                  <div class="border-t border-slate-200 pt-6">
                    <p class="text-xs text-slate-500 mb-2">Full Transaction ID</p>
                    <div class="bg-slate-50 rounded-lg p-3 font-mono text-sm text-slate-700 break-all">
                      {transaction.id}
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div class="bg-slate-50 border-t border-slate-200 px-8 py-6">
                  <div class="flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div class="text-center sm:text-left">
                      <p class="text-sm text-slate-600">
                        Questions about this transaction?
                      </p>
                      <div class="flex flex-wrap gap-4 mt-2">
                        <a
                          href="mailto:support@ciftmarkets.com"
                          class="text-sm text-blue-500 hover:text-blue-600 font-medium"
                        >
                          support@ciftmarkets.com
                        </a>
                        <a
                          href="tel:+16469782187"
                          class="text-sm text-blue-500 hover:text-blue-600 font-medium"
                        >
                          +1 (646) 978-2187
                        </a>
                      </div>
                    </div>
                    <button
                      onClick={() => navigate('/')}
                      class="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
                    >
                      <ArrowLeft class="w-4 h-4" />
                      Return to Home
                    </button>
                  </div>
                </div>
              </div>
            );
          }}
        </Show>
      </div>

      {/* Footer */}
      <div class="max-w-4xl mx-auto px-4 py-8 mt-12 border-t border-slate-200">
        <p class="text-center text-sm text-slate-500">
          © {new Date().getFullYear()} CIFT Markets. Member FINRA/SIPC. All rights reserved.
        </p>
      </div>
    </div>
  );
};

export default VerifyTransactionPage;
