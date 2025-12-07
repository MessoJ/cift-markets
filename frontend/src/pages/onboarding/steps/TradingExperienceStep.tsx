import { createSignal, createEffect } from 'solid-js';
import { KYCProfile } from '../../../lib/api/client';

interface TradingExperienceStepProps {
  profile: Partial<KYCProfile>;
  onComplete: (data: Partial<KYCProfile>) => void;
}

export function TradingExperienceStep(props: TradingExperienceStepProps) {
  const [stocksExp, setStocksExp] = createSignal(props.profile.trading_experience?.stocks || 'none');
  const [optionsExp, setOptionsExp] = createSignal(props.profile.trading_experience?.options || 'none');
  const [marginExp, setMarginExp] = createSignal(props.profile.trading_experience?.margin || 'none');
  const [isPEP, setIsPEP] = createSignal(props.profile.is_politically_exposed || false);
  const [isAffiliated, setIsAffiliated] = createSignal(props.profile.is_affiliated_with_exchange || false);
  const [isControl, setIsControl] = createSignal(props.profile.is_control_person || false);

  createEffect(() => {
    props.onComplete({
      trading_experience: {
        stocks: stocksExp() as any,
        options: optionsExp() as any,
        margin: marginExp() as any,
      },
      is_politically_exposed: isPEP(),
      is_affiliated_with_exchange: isAffiliated(),
      is_control_person: isControl(),
    });
  });

  const ExperienceSelector = (props: {
    label: string;
    value: string;
    onChange: (val: string) => void;
  }) => (
    <div>
      <label class="text-xs text-gray-400 block mb-2">{props.label} *</label>
      <div class="grid grid-cols-4 gap-2">
        {[
          { value: 'none', label: 'None' },
          { value: 'limited', label: 'Limited' },
          { value: 'good', label: 'Good' },
          { value: 'extensive', label: 'Extensive' },
        ].map((opt) => (
          <button
            onClick={() => props.onChange(opt.value)}
            class={`px-4 py-2 rounded border text-sm font-semibold transition-colors ${
              props.value === opt.value
                ? 'bg-accent-500/10 border-accent-500 text-accent-500'
                : 'bg-terminal-850 border-terminal-750 text-gray-400 hover:border-gray-600'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div>
      <h2 class="text-lg sm:text-xl font-bold text-white mb-2">Trading Experience</h2>
      <p class="text-xs sm:text-sm text-gray-400 mb-4 sm:mb-6">Tell us about your trading experience and regulatory status</p>

      <div class="space-y-4 sm:space-y-6">
        <div class="space-y-3 sm:space-y-4">
          <ExperienceSelector
            label="Stocks & ETFs Experience"
            value={stocksExp()}
            onChange={setStocksExp}
          />
          <ExperienceSelector
            label="Options Trading Experience"
            value={optionsExp()}
            onChange={setOptionsExp}
          />
          <ExperienceSelector
            label="Margin Trading Experience"
            value={marginExp()}
            onChange={setMarginExp}
          />
        </div>

        <div class="p-3 sm:p-4 bg-terminal-850 border border-terminal-750 rounded">
          <div class="text-xs sm:text-sm font-semibold text-white mb-3 sm:mb-4">Regulatory Disclosures</div>
          <div class="space-y-2 sm:space-y-3">
            <label class="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={isPEP()}
                onChange={(e) => setIsPEP(e.target.checked)}
                class="mt-0.5 w-4 h-4 bg-terminal-900 border-terminal-750 rounded"
              />
              <div class="text-xs sm:text-sm text-gray-400 group-hover:text-white transition-colors">
                I am a <span class="font-semibold text-white">Politically Exposed Person (PEP)</span> or
                senior foreign political figure
              </div>
            </label>

            <label class="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={isAffiliated()}
                onChange={(e) => setIsAffiliated(e.target.checked)}
                class="mt-0.5 w-4 h-4 bg-terminal-900 border-terminal-750 rounded"
              />
              <div class="text-xs sm:text-sm text-gray-400 group-hover:text-white transition-colors">
                I am <span class="font-semibold text-white">affiliated with a stock exchange, FINRA member,
                or publicly traded company</span>
              </div>
            </label>

            <label class="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={isControl()}
                onChange={(e) => setIsControl(e.target.checked)}
                class="mt-0.5 w-4 h-4 bg-terminal-900 border-terminal-750 rounded"
              />
              <div class="text-xs sm:text-sm text-gray-400 group-hover:text-white transition-colors">
                I am a <span class="font-semibold text-white">control person, director, or 10% shareholder</span> of
                a publicly traded company
              </div>
            </label>
          </div>
        </div>

        <div class="p-3 sm:p-4 bg-primary-500/5 border border-primary-500/20 rounded text-[10px] sm:text-xs text-gray-400">
          <span class="font-semibold text-primary-500">Experience Definitions:</span>
          <ul class="mt-2 space-y-1 list-disc list-inside">
            <li><strong>None:</strong> No prior trading experience</li>
            <li><strong>Limited:</strong> 1-2 years, occasional trading</li>
            <li><strong>Good:</strong> 2-5 years, regular trading activity</li>
            <li><strong>Extensive:</strong> 5+ years, frequent and sophisticated trading</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
