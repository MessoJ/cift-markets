import { createSignal, createEffect } from 'solid-js';
import { FileText } from 'lucide-solid';
import { KYCProfile } from '../../../lib/api/client';

interface AgreementsStepProps {
  profile: Partial<KYCProfile>;
  onComplete: (data: Partial<KYCProfile>) => void;
}

export function AgreementsStep(props: AgreementsStepProps) {
  const [customerAgreement, setCustomerAgreement] = createSignal(
    props.profile.agreements_accepted?.customer_agreement || false
  );
  const [marginAgreement, setMarginAgreement] = createSignal(
    props.profile.agreements_accepted?.margin_agreement || false
  );
  const [optionsAgreement, setOptionsAgreement] = createSignal(
    props.profile.agreements_accepted?.options_agreement || false
  );
  const [electronicDelivery, setElectronicDelivery] = createSignal(
    props.profile.agreements_accepted?.electronic_delivery || false
  );

  createEffect(() => {
    if (customerAgreement() && electronicDelivery()) {
      props.onComplete({
        agreements_accepted: {
          customer_agreement: customerAgreement(),
          margin_agreement: marginAgreement(),
          options_agreement: optionsAgreement(),
          electronic_delivery: electronicDelivery(),
        },
      });
    }
  });

  return (
    <div>
      <h2 class="text-lg sm:text-xl font-bold text-white mb-2">Legal Agreements</h2>
      <p class="text-xs sm:text-sm text-gray-400 mb-4 sm:mb-6">Review and accept required agreements</p>

      <div class="space-y-3 sm:space-y-4">
        {/* Customer Agreement */}
        <div class="p-3 sm:p-4 bg-terminal-850 border border-terminal-750 rounded">
          <label class="flex items-start gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={customerAgreement()}
              onChange={(e) => setCustomerAgreement(e.target.checked)}
              class="mt-1 w-5 h-5 bg-terminal-900 border-terminal-750 rounded"
            />
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2">
                <FileText size={14} class="sm:w-4 sm:h-4 text-accent-500" />
                <span class="text-xs sm:text-sm font-semibold text-white">
                  Customer Agreement *
                </span>
              </div>
              <div class="text-xs text-gray-400 mb-2">
                I have read and agree to the CIFT Markets Customer Agreement, which governs the
                terms of my brokerage account, including trading rules, fees, and dispute resolution.
              </div>
              <a
                href="/legal/customer-agreement"
                target="_blank"
                class="text-[10px] sm:text-xs text-accent-500 hover:underline"
              >
                View Agreement →
              </a>
            </div>
          </label>
        </div>

        {/* Margin Agreement */}
        <div class="p-3 sm:p-4 bg-terminal-850 border border-terminal-750 rounded">
          <label class="flex items-start gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={marginAgreement()}
              onChange={(e) => setMarginAgreement(e.target.checked)}
              class="mt-1 w-5 h-5 bg-terminal-900 border-terminal-750 rounded"
            />
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2">
                <FileText size={14} class="sm:w-4 sm:h-4 text-warning-500" />
                <span class="text-xs sm:text-sm font-semibold text-white">
                  Margin Agreement (Optional)
                </span>
              </div>
              <div class="text-xs text-gray-400 mb-2">
                I understand the risks of margin trading and agree to the terms outlined in the
                Margin Disclosure Statement, including the potential for unlimited losses.
              </div>
              <a
                href="/legal/margin-agreement"
                target="_blank"
                class="text-xs text-accent-500 hover:underline"
              >
                View Agreement →
              </a>
            </div>
          </label>
        </div>

        {/* Options Agreement */}
        <div class="p-3 sm:p-4 bg-terminal-850 border border-terminal-750 rounded">
          <label class="flex items-start gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={optionsAgreement()}
              onChange={(e) => setOptionsAgreement(e.target.checked)}
              class="mt-1 w-5 h-5 bg-terminal-900 border-terminal-750 rounded"
            />
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2">
                <FileText size={14} class="sm:w-4 sm:h-4 text-primary-500" />
                <span class="text-sm font-semibold text-white">
                  Options Agreement (Optional)
                </span>
              </div>
              <div class="text-xs text-gray-400 mb-2">
                I have read the Options Disclosure Document and understand the risks associated
                with options trading, including complete loss of investment.
              </div>
              <a
                href="/legal/options-agreement"
                target="_blank"
                class="text-xs text-accent-500 hover:underline"
              >
                View Agreement →
              </a>
            </div>
          </label>
        </div>

        {/* Electronic Delivery */}
        <div class="p-4 bg-terminal-850 border border-terminal-750 rounded">
          <label class="flex items-start gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={electronicDelivery()}
              onChange={(e) => setElectronicDelivery(e.target.checked)}
              class="mt-1 w-5 h-5 bg-terminal-900 border-terminal-750 rounded"
            />
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2">
                <FileText size={16} class="text-success-500" />
                <span class="text-sm font-semibold text-white">
                  Electronic Delivery Consent *
                </span>
              </div>
              <div class="text-xs text-gray-400">
                I consent to receive all account statements, confirmations, tax documents, and
                regulatory disclosures electronically via email and the CIFT Markets platform.
              </div>
            </div>
          </label>
        </div>

        <div class="p-4 bg-danger-500/5 border border-danger-500/20 rounded">
          <div class="text-sm font-semibold text-danger-500 mb-2">Important Legal Disclosures</div>
          <div class="text-xs text-gray-400 space-y-2">
            <p>
              <strong>Risk Warning:</strong> Trading securities involves substantial risk and may not be
              suitable for all investors. You may lose all or more than your initial investment.
            </p>
            <p>
              <strong>No Investment Advice:</strong> CIFT Markets does not provide investment advice,
              tax advice, or legal advice. Consult your own advisors before making investment decisions.
            </p>
            <p>
              <strong>SIPC Protection:</strong> Securities in your account are protected up to $500,000
              by the Securities Investor Protection Corporation (SIPC), including $250,000 for cash claims.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
