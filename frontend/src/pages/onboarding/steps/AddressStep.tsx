import { createSignal, createEffect } from 'solid-js';
import { KYCProfile } from '../../../lib/api/client';

interface AddressStepProps {
  profile: Partial<KYCProfile>;
  onComplete: (data: Partial<KYCProfile>) => void;
}

export function AddressStep(props: AddressStepProps) {
  const [addressLine1, setAddressLine1] = createSignal(props.profile.address_line1 || '');
  const [addressLine2, setAddressLine2] = createSignal(props.profile.address_line2 || '');
  const [city, setCity] = createSignal(props.profile.city || '');
  const [state, setState] = createSignal(props.profile.state || '');
  const [postalCode, setPostalCode] = createSignal(props.profile.postal_code || '');
  const [country, setCountry] = createSignal(props.profile.country || 'US');

  createEffect(() => {
    if (addressLine1() && city() && state() && postalCode() && country()) {
      props.onComplete({
        address_line1: addressLine1(),
        address_line2: addressLine2() || undefined,
        city: city(),
        state: state(),
        postal_code: postalCode(),
        country: country(),
      });
    }
  });

  return (
    <div>
      <h2 class="text-lg sm:text-xl font-bold text-white mb-2">Residential Address</h2>
      <p class="text-xs sm:text-sm text-gray-400 mb-4 sm:mb-6">We need your physical address for verification</p>

      <div class="space-y-3 sm:space-y-4">
        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Country *</label>
          <select
            value={country()}
            onChange={(e) => setCountry(e.target.value)}
            class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
          >
            <option value="US">United States</option>
            <option value="CA">Canada</option>
            <option value="GB">United Kingdom</option>
          </select>
        </div>

        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Address Line 1 *</label>
          <input
            type="text"
            value={addressLine1()}
            onInput={(e) => setAddressLine1(e.target.value)}
            placeholder="123 Main Street"
            class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
          />
        </div>

        <div>
          <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">Address Line 2</label>
          <input
            type="text"
            value={addressLine2()}
            onInput={(e) => setAddressLine2(e.target.value)}
            placeholder="Apt 4B (optional)"
            class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
          />
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
          <div class="col-span-1">
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">City *</label>
            <input
              type="text"
              value={city()}
              onInput={(e) => setCity(e.target.value)}
              placeholder="New York"
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
          <div class="col-span-1">
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">State *</label>
            <input
              type="text"
              value={state()}
              onInput={(e) => setState(e.target.value)}
              placeholder="NY"
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
          <div class="col-span-1">
            <label class="text-xs text-gray-400 block mb-1.5 sm:mb-2">ZIP Code *</label>
            <input
              type="text"
              value={postalCode()}
              onInput={(e) => setPostalCode(e.target.value)}
              placeholder="10001"
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base rounded focus:outline-none focus:border-accent-500"
            />
          </div>
        </div>

        <div class="p-3 sm:p-4 bg-warning-500/5 border border-warning-500/20 rounded text-[10px] sm:text-xs text-gray-400">
          <span class="font-semibold text-warning-500">Important:</span> P.O. Boxes are not accepted.
          You must provide a physical residential address.
        </div>
      </div>
    </div>
  );
}
