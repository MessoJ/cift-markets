import { createSignal, createEffect, Show } from 'solid-js';
import { KYCProfile } from '../../../lib/api/client';

interface EmploymentStepProps {
  profile: Partial<KYCProfile>;
  onComplete: (data: Partial<KYCProfile>) => void;
}

export function EmploymentStep(props: EmploymentStepProps) {
  const [employmentStatus, setEmploymentStatus] = createSignal(props.profile.employment_status || 'employed');
  const [employerName, setEmployerName] = createSignal(props.profile.employer_name || '');
  const [occupation, setOccupation] = createSignal(props.profile.occupation || '');
  const [annualIncome, setAnnualIncome] = createSignal(props.profile.annual_income_range || '');
  const [netWorth, setNetWorth] = createSignal(props.profile.net_worth_range || '');
  const [riskTolerance, setRiskTolerance] = createSignal(props.profile.risk_tolerance || 'moderate');
  const [objectives, setObjectives] = createSignal<string[]>(props.profile.investment_objectives || []);

  const toggleObjective = (obj: string) => {
    const current = objectives();
    if (current.includes(obj)) {
      setObjectives(current.filter((o) => o !== obj));
    } else {
      setObjectives([...current, obj]);
    }
  };

  createEffect(() => {
    const needsEmployer = employmentStatus() === 'employed' || employmentStatus() === 'self_employed';
    const employerValid = needsEmployer ? employerName().length > 0 : true;
    
    if (employmentStatus() && annualIncome() && netWorth() && riskTolerance() && objectives().length > 0 && employerValid) {
      props.onComplete({
        employment_status: employmentStatus() as any,
        employer_name: employerName() || undefined,
        occupation: occupation() || undefined,
        annual_income_range: annualIncome(),
        net_worth_range: netWorth(),
        risk_tolerance: riskTolerance() as any,
        investment_objectives: objectives(),
      });
    }
  });

  return (
    <div>
      <h2 class="text-lg sm:text-xl font-bold text-white mb-2">Employment & Financial Information</h2>
      <p class="text-xs sm:text-sm text-gray-400 mb-4 sm:mb-6">Help us understand your financial situation</p>

      <div class="space-y-3 sm:space-y-4">
        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Employment Status *</label>
          <select
            value={employmentStatus()}
            onChange={(e) => setEmploymentStatus(e.target.value)}
            class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
          >
            <option value="employed">Employed</option>
            <option value="self_employed">Self-Employed</option>
            <option value="retired">Retired</option>
            <option value="student">Student</option>
            <option value="unemployed">Unemployed</option>
          </select>
        </div>

        <Show when={employmentStatus() === 'employed' || employmentStatus() === 'self_employed'}>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <div>
              <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Employer Name *</label>
              <input
                type="text"
                value={employerName()}
                onInput={(e) => setEmployerName(e.target.value)}
                placeholder="Company Name"
                class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
              />
            </div>
            <div>
              <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Occupation</label>
              <input
                type="text"
                value={occupation()}
                onInput={(e) => setOccupation(e.target.value)}
                placeholder="Software Engineer"
                class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
              />
            </div>
          </div>
        </Show>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
          <div>
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Annual Income *</label>
            <select
              value={annualIncome()}
              onChange={(e) => setAnnualIncome(e.target.value)}
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            >
              <option value="">Select range...</option>
              <option value="0-25k">$0 - $25,000</option>
              <option value="25k-50k">$25,000 - $50,000</option>
              <option value="50k-100k">$50,000 - $100,000</option>
              <option value="100k-200k">$100,000 - $200,000</option>
              <option value="200k-500k">$200,000 - $500,000</option>
              <option value="500k+">$500,000+</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Net Worth *</label>
            <select
              value={netWorth()}
              onChange={(e) => setNetWorth(e.target.value)}
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            >
              <option value="">Select range...</option>
              <option value="0-50k">$0 - $50,000</option>
              <option value="50k-100k">$50,000 - $100,000</option>
              <option value="100k-250k">$100,000 - $250,000</option>
              <option value="250k-500k">$250,000 - $500,000</option>
              <option value="500k-1m">$500,000 - $1,000,000</option>
              <option value="1m+">$1,000,000+</option>
            </select>
          </div>
        </div>

        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Investment Objectives * (Select all that apply)</label>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {[
              { id: 'growth', label: 'Capital Growth' },
              { id: 'income', label: 'Income Generation' },
              { id: 'preservation', label: 'Capital Preservation' },
              { id: 'speculation', label: 'Speculation' },
              { id: 'trading', label: 'Active Trading' },
              { id: 'hedging', label: 'Hedging' },
            ].map((obj) => (
              <button
                onClick={() => toggleObjective(obj.id)}
                class={`p-2 sm:p-3 rounded border transition-colors text-left ${
                  objectives().includes(obj.id)
                    ? 'bg-accent-500/10 border-accent-500 text-accent-500'
                    : 'bg-terminal-850 border-terminal-750 text-gray-400 hover:border-gray-600'
                }`}
              >
                <div class="text-xs sm:text-sm font-semibold">{obj.label}</div>
              </button>
            ))}
          </div>
        </div>

        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Risk Tolerance *</label>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-3">
            <button
              onClick={() => setRiskTolerance('conservative')}
              class={`p-3 sm:p-4 rounded border transition-colors ${
                riskTolerance() === 'conservative'
                  ? 'bg-success-500/10 border-success-500 text-success-500'
                  : 'bg-terminal-850 border-terminal-750 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div class="text-xs sm:text-sm font-semibold mb-1">Conservative</div>
              <div class="text-[10px] sm:text-xs text-gray-500">Low risk, stable returns</div>
            </button>
            <button
              onClick={() => setRiskTolerance('moderate')}
              class={`p-3 sm:p-4 rounded border transition-colors ${
                riskTolerance() === 'moderate'
                  ? 'bg-warning-500/10 border-warning-500 text-warning-500'
                  : 'bg-terminal-850 border-terminal-750 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div class="text-xs sm:text-sm font-semibold mb-1">Moderate</div>
              <div class="text-[10px] sm:text-xs text-gray-500">Balanced risk/reward</div>
            </button>
            <button
              onClick={() => setRiskTolerance('aggressive')}
              class={`p-3 sm:p-4 rounded border transition-colors ${
                riskTolerance() === 'aggressive'
                  ? 'bg-danger-500/10 border-danger-500 text-danger-500'
                  : 'bg-terminal-850 border-terminal-750 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div class="text-xs sm:text-sm font-semibold mb-1">Aggressive</div>
              <div class="text-[10px] sm:text-xs text-gray-500">High risk, high reward</div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
