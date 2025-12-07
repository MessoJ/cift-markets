import { Component } from 'solid-js';

interface PaymentMethodLogoProps {
  type: 'visa' | 'mastercard' | 'amex' | 'discover' | 'paypal' | 'bitcoin' | 'ethereum' | 'bank' | 'mpesa' | 'crypto' | 'cashapp';
  class?: string;
  size?: number;
}

/**
 * Payment Method Logo Component
 * Displays authentic payment method logos using SVG
 */
export const PaymentMethodLogo: Component<PaymentMethodLogoProps> = (props) => {
  const size = props.size || 32;
  const defaultClass = props.class || '';

  // Visa Logo - Official colors
  const VisaLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="Visa">
      <rect width="48" height="32" rx="4" fill="#1434CB"/>
      <g transform="translate(8, 9)">
        <path d="M8.5 12l1.8-11h2.8l-1.8 11h-2.8zm14-10.8c-.6-.2-1.5-.5-2.6-.5-2.8 0-4.8 1.5-4.8 3.6-.1 1.6 1.4 2.4 2.5 2.9 1.2.5 1.6.8 1.6 1.3 0 .7-.8 1.1-1.7 1.1-1.1 0-1.8-.2-2.7-.6l-.4-.2-.4 2.4c.7.3 1.9.6 3.2.6 3 0 5-1.5 5-3.7 0-1.2-.7-2.2-2.4-2.9-1-.5-1.6-.8-1.6-1.3 0-.5.5-.9 1.6-.9.8 0 1.5.2 2 .4l.2.1.4-2.3zM31.8 1h-2.3c-.7 0-1.2.2-1.5 1l-4.3 10.2h3l.6-1.7h3.7l.4 1.7h2.6L31.8 1zm-3.4 7.2c.2-.6 1.1-3.1 1.1-3.1l.6 3.1h-2.4l.7 0zM11.5 1L8.7 8.6l-.3-1.5C7.9 5.5 6.3 3.7 4.5 2.8l2.6 9.6h3L15.8 1h-4.3z" fill="white"/>
        <path d="M6 1H.8l-.1.4c3.6 1 6 3.1 7 5.8L6.8 2.7C6.6 1.9 6 1.5 6 1z" fill="#F7B600"/>
      </g>
    </svg>
  );

  // Mastercard Logo
  const MastercardLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="Mastercard">
      <rect width="48" height="32" rx="4" fill="#000000"/>
      <circle cx="19" cy="16" r="9" fill="#EB001B"/>
      <circle cx="29" cy="16" r="9" fill="#F79E1B"/>
      <path d="M24 9.5c-1.7 1.5-2.8 3.6-2.8 6s1.1 4.5 2.8 6c1.7-1.5 2.8-3.6 2.8-6s-1.1-4.5-2.8-6z" fill="#FF5F00"/>
    </svg>
  );

  // American Express Logo
  const AmexLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="American Express">
      <rect width="48" height="32" rx="4" fill="#006FCF"/>
      <path d="M10 12.5h3.5l.8 1.8.8-1.8h3.5v5.4l2.5-5.4h3l2.5 5.4v-5.4h3.5l.7 1.6.7-1.6h11v1.3l1-.6h5.5l-2.5 5.7h-3.3l-.3-.7h-2l-.3.7h-3.6l-1.3-1.5-1.3 1.5h-7.5v-1.3l-.5 1.3h-2.9l-.5-1.3v1.3h-9.8l-.7-1.6-.7 1.6h-3.9l2.5-5.4h-3.5v-1.5zm20 1.3l-1.5 3.4h1l.2-.5h1.2l.2.5h1l-1.5-3.4h-.6zm-15.5 0v3.4h2l1.3-1.5v1.5h2v-3.4h-2l-1.3 1.5v-1.5h-2zm10 0l-1.3 3.4h.9l.2-.5h1.2l.2.5h1l-1.3-3.4h-.9zm6 0v.8h1.5v2.6h1v-2.6h1.5v-.8h-4zm-10 .9l.4 1h-.8l.4-1z" fill="white"/>
    </svg>
  );

  // Discover Logo
  const DiscoverLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="Discover">
      <rect width="48" height="32" rx="4" fill="#FF6000"/>
      <path d="M35 10h8c2.2 0 4 1.8 4 4v4c0 2.2-1.8 4-4 4h-8c-2.2 0-4-1.8-4-4v-4c0-2.2 1.8-4 4-4z" fill="#F47216"/>
      <text x="8" y="20" font-family="Arial, sans-serif" font-size="7" font-weight="bold" fill="white">DISCOVER</text>
    </svg>
  );

  // PayPal Logo - Official icon design
  const PayPalLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="PayPal">
      <rect width="48" height="32" rx="4" fill="#003087"/>
      <g transform="translate(12, 4)">
        <path d="M5 2h8c3 0 5 1.5 5 4.5 0 3.5-2.5 6-6 6H9l-1.5 7H4L5 2z" fill="#9FC8E8" fill-opacity="0.7"/>
        <path d="M8 5h6c2 0 3.5 1 3.5 3 0 2.5-1.8 4.5-4.5 4.5h-3l-1 5h-2.5L8 5z" fill="white"/>
      </g>
    </svg>
  );

  // Bitcoin Logo
  const BitcoinLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width={size} height={size} class={defaultClass} role="img" aria-label="Bitcoin">
      <circle cx="16" cy="16" r="16" fill="#F7931A"/>
      <path d="M23.2 14.3c.3-2-1.2-3.1-3.3-3.8l.7-2.7-1.7-.4-.6 2.6c-.4-.1-.9-.2-1.4-.3l.7-2.6-1.7-.4-.7 2.7c-.4-.1-.7-.2-1.1-.2v-.1l-2.3-.6-.4 1.8s1.2.3 1.2.3c.7.2.8.6.8 1l-.8 3.2c0 .1.1.1.2.2h-.2l-1.1 4.4c-.1.2-.3.5-.8.4 0 0-1.2-.3-1.2-.3l-.9 1.9 2.2.5c.4.1.8.2 1.2.3l-.7 2.8 1.7.4.7-2.7c.5.1.9.2 1.4.3l-.7 2.7 1.7.4.7-2.8c2.9.5 5.1.3 6-2.3.7-2.1-.1-3.3-1.5-4.1 1.1-.3 1.9-1 2.1-2.6zm-3.8 5.4c-.5 2.1-4 1-5.1.7l.9-3.6c1.2.3 4.8.9 4.2 2.9zm.5-5.4c-.5 1.9-3.4.9-4.4.7l.8-3.3c1 .2 4 .7 3.6 2.6z" fill="white"/>
    </svg>
  );

  // Ethereum Logo
  const EthereumLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width={size} height={size} class={defaultClass} role="img" aria-label="Ethereum">
      <circle cx="16" cy="16" r="16" fill="#627EEA"/>
      <g fill="white" fill-opacity=".8">
        <path d="M16 4l-.3.9v13.3l.3.3 6-3.5z"/>
        <path d="M16 4l-6 10.9 6 3.5z"/>
        <path d="M16 20.5l-.2.2v4.6l.2.6 6-8.4z"/>
        <path d="M16 25.9v-5.4l-6-3.5z"/>
        <path d="M16 18.4l6-3.5-6-2.7z"/>
        <path d="M10 14.9l6 3.5v-6.2z"/>
      </g>
    </svg>
  );

  // Bank Account Logo
  const BankLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width={size} height={size} class={defaultClass} role="img" aria-label="Bank Account">
      <circle cx="16" cy="16" r="16" fill="#3B82F6"/>
      <path d="M16 6l10 6v2H6v-2l10-6zm-8 10h2v6h-2v-6zm4 0h2v6h-2v-6zm4 0h2v6h-2v-6zm4 0h2v6h-2v-6zm-10 8h16v2H10v-2z" fill="white"/>
    </svg>
  );

  // M-Pesa Logo - Official design with phone icon
  const MpesaLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="M-Pesa">
      <rect width="48" height="32" rx="4" fill="#00D64F"/>
      <g transform="translate(3, 9)">
        <rect x="8" y="2" width="4" height="8" rx="0.5" fill="#8BC34A" fill-opacity="0.6"/>
        <path d="M9 5l2 1.5" stroke="#E53935" stroke-width="0.8" fill="none" stroke-linecap="round"/>
        <text x="0" y="9" font-family="Arial, Helvetica, sans-serif" font-size="5.5" font-weight="bold" fill="white" letter-spacing="0.3">M</text>
        <text x="13" y="9" font-family="Arial, Helvetica, sans-serif" font-size="5.5" font-weight="bold" fill="white" letter-spacing="0.5">PESA</text>
      </g>
    </svg>
  );

  // Generic Crypto Logo
  const CryptoLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width={size} height={size} class={defaultClass} role="img" aria-label="Cryptocurrency">
      <defs>
        <linearGradient id="cryptoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#F7931A;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#627EEA;stop-opacity:1" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="16" fill="url(#cryptoGrad)"/>
      <path d="M16 6l8 5v10l-8 5-8-5V11l8-5zm0 2l-6 3.8v7.6l6 3.8 6-3.8v-7.6L16 8zm0 3.5c2.5 0 4.5 2 4.5 4.5s-2 4.5-4.5 4.5-4.5-2-4.5-4.5 2-4.5 4.5-4.5zm0 1.5c-1.7 0-3 1.3-3 3s1.3 3 3 3 3-1.3 3-3-1.3-3-3-3z" fill="white"/>
    </svg>
  );

  // Cash App Logo - Official design
  const CashAppLogo = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 32" width={size} height={size * 0.67} class={defaultClass} role="img" aria-label="Cash App">
      <rect width="48" height="32" rx="4" fill="#00D632"/>
      <g transform="translate(14, 6)">
        <path d="M12 2c-1.5 0-2.8.5-3.8 1.3l-1-2.8L5 1.2l1.2 3.3C5.5 5.5 5 6.8 5 8.2c0 2.8 1.8 5.2 4.3 6.2l1.2 3.3 2.2-.7-1-2.8c1-.5 1.8-1.2 2.5-2.1l1.8 5 2.2-.7-1.8-5c.5-1 .8-2.2.8-3.4 0-1.5-.5-2.8-1.3-3.9l1-2.8-2.2-.7-1 2.8c-.8-.4-1.7-.7-2.7-.7zm0 2c1.7 0 3 1.3 3 3s-1.3 3-3 3-3-1.3-3-3 1.3-3 3-3z" fill="white"/>
        <circle cx="12" cy="8" r="2" fill="#00D632"/>
      </g>
    </svg>
  );

  const logos = {
    visa: <VisaLogo />,
    mastercard: <MastercardLogo />,
    amex: <AmexLogo />,
    discover: <DiscoverLogo />,
    paypal: <PayPalLogo />,
    bitcoin: <BitcoinLogo />,
    ethereum: <EthereumLogo />,
    bank: <BankLogo />,
    mpesa: <MpesaLogo />,
    crypto: <CryptoLogo />,
    cashapp: <CashAppLogo />,
  };

  return <>{logos[props.type] || <BankLogo />}</>;
};
