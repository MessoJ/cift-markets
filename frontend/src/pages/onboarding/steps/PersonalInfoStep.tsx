import { createSignal, createEffect } from 'solid-js';
import { KYCProfile } from '../../../lib/api/client';

interface PersonalInfoStepProps {
  profile: Partial<KYCProfile>;
  onComplete: (data: Partial<KYCProfile>) => void;
}

export function PersonalInfoStep(props: PersonalInfoStepProps) {
  const [firstName, setFirstName] = createSignal(props.profile.first_name || '');
  const [middleName, setMiddleName] = createSignal(props.profile.middle_name || '');
  const [lastName, setLastName] = createSignal(props.profile.last_name || '');
  const [dateOfBirth, setDateOfBirth] = createSignal(props.profile.date_of_birth || '');
  const [phone, setPhone] = createSignal(props.profile.phone || '');
  const [accountType, setAccountType] = createSignal(props.profile.account_type || 'individual');

  createEffect(() => {
    if (firstName() && lastName() && dateOfBirth() && phone()) {
      props.onComplete({
        first_name: firstName(),
        middle_name: middleName() || undefined,
        last_name: lastName(),
        date_of_birth: dateOfBirth(),
        phone: phone(),
        account_type: accountType() as any,
      });
    }
  });

  return (
    <div>
      <h2 class="text-lg sm:text-xl font-bold text-white mb-2">Personal Information</h2>
      <p class="text-xs sm:text-sm text-gray-400 mb-4 sm:mb-6">Tell us about yourself</p>

      <div class="space-y-3 sm:space-y-4">
        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Account Type *</label>
          <select
            value={accountType()}
            onChange={(e) => setAccountType(e.target.value)}
            class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
          >
            <option value="individual">Individual</option>
            <option value="joint">Joint</option>
            <option value="ira">IRA</option>
            <option value="trust">Trust</option>
            <option value="business">Business</option>
          </select>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
          <div class="col-span-1">
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">First Name *</label>
            <input
              type="text"
              value={firstName()}
              onInput={(e) => setFirstName(e.target.value)}
              placeholder="John"
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
          <div class="col-span-1">
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Middle Name</label>
            <input
              type="text"
              value={middleName()}
              onInput={(e) => setMiddleName(e.target.value)}
              placeholder="M."
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
          <div class="col-span-1">
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Last Name *</label>
            <input
              type="text"
              value={lastName()}
              onInput={(e) => setLastName(e.target.value)}
              placeholder="Doe"
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
          <div>
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Date of Birth *</label>
            <input
              type="date"
              value={dateOfBirth()}
              onInput={(e) => setDateOfBirth(e.target.value)}
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
          <div>
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Phone Number *</label>
            <input
              type="tel"
              value={phone()}
              onInput={(e) => setPhone(e.target.value)}
              placeholder="+1 (555) 123-4567"
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded focus:outline-none focus:border-accent-500"
            />
          </div>
        </div>

        <div class="p-4 bg-primary-500/5 border border-primary-500/20 rounded text-xs text-gray-400">
          <span class="font-semibold text-primary-500">Why we need this:</span> Federal regulations require
          us to verify your identity. Your information is encrypted and securely stored.
        </div>
      </div>
    </div>
  );
}
