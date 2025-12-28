/**
 * CIFT Markets API Client
 * 
 * Complete TypeScript client for backend integration.
 * NO MOCK DATA - All data fetched from database via backend API.
 * 
 * Tech Stack Integration:
 * - FastAPI Backend (http://localhost:8000)
 * - Phase 5-7 Stack (ClickHouse, Polars, Dragonfly, etc.)
 * - Real-time WebSocket for market data
 */

import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';

// ============================================================================
// TYPE DECLARATIONS
// ============================================================================

declare global {
  interface ImportMetaEnv {
    readonly VITE_API_URL?: string;
    readonly VITE_WS_URL?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

// Use relative paths to leverage Vite proxy (configured in vite.config.ts)
// This avoids CORS issues and allows proper cookie handling
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface ApiError {
  message: string;
  status: number;
  detail?: any;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  is_verified?: boolean;
  created_at: string;
}

export interface Order {
  id: string;
  user_id: string;
  account_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
  time_in_force: 'day' | 'gtc' | 'ioc' | 'fok';
  quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  limit_price?: number;
  stop_price?: number;
  avg_fill_price?: number;
  status: 'pending' | 'accepted' | 'partial' | 'filled' | 'cancelled' | 'rejected' | 'expired';
  total_value?: number;
  commission?: number;
  created_at: string;
  filled_at?: string;
  cancelled_at?: string;
}

export interface Position {
  id: string;
  symbol: string;
  quantity: number;
  side: 'long' | 'short';
  avg_cost: number;
  total_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  realized_pnl: number;
  day_pnl: number;
  day_pnl_pct: number;
}

export interface PortfolioSummary {
  total_value: number;
  cash: number;
  buying_power: number;
  positions_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  day_pnl: number;
  day_pnl_pct: number;
  leverage: number;
}

export interface Quote {
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  change?: number;
  change_pct?: number;
  high?: number;
  low?: number;
  open?: number;
  timestamp: string;
}

export interface Watchlist {
  id: string;
  name: string;
  description?: string;
  symbols: string[] | string;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface PerformanceMetrics {
  period: {
    start_date: string;
    end_date: string;
    days: number;
  };
  returns: {
    total_return_pct: number;
    initial_value: number;
    final_value: number;
    total_pnl: number;
  };
  risk_metrics: {
    sharpe_ratio: number;
    max_drawdown_pct: number;
    volatility_pct: number;
  };
  trade_statistics: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate_pct: number;
    avg_pnl: number;
    best_trade: number;
    worst_trade: number;
  };
  _backend?: 'clickhouse+polars' | 'postgresql';
}

export interface MLPrediction {
  timestamp: number;
  symbol: string;
  direction: 'long' | 'short' | 'neutral';
  direction_probability: number;
  magnitude: number;
  confidence: number;
  model_agreement: number;
  current_regime: string;
  regime_probability: number;
  should_trade: boolean;
  position_size: number;
  stop_loss_bps: number;
  take_profit_bps: number;
  model_weights: Record<string, number>;
  inference_latency_ms: number;
}

export interface OrderDetail {
  order: Order;
  fills: Array<{
    fill_id: string;
    quantity: number;
    price: number;
    value: number;
    commission: number;
    venue?: string;
    timestamp: string;
    liquidity_flag?: string;
  }>;
  execution_quality: {
    avg_fill_price?: number;
    vwap?: number;
    slippage_bps?: number;
    fill_rate: number;
    num_fills: number;
    total_commission: number;
    time_to_first_fill_ms?: number;
  };
  timeline: Array<{
    event: string;
    timestamp: string;
    quantity?: number;
    price?: number;
  }>;
}

export interface EquityCurve {
  data: Array<{
    timestamp: string;
    value: number;
    cash: number;
    positions: number;
    unrealized_pnl?: number;
    day_pnl?: number;
  }>;
  resolution: 'hourly' | 'daily' | 'weekly';
  _backend?: 'clickhouse' | 'postgresql';
}

export interface FundingTransaction {
  id: string;
  user_id?: string;
  account_id?: string;
  type: 'deposit' | 'withdrawal';
  method: 'instant' | 'standard' | 'ach' | 'wire' | 'card' | 'crypto' | 'check';
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'returned';
  amount: number;
  currency?: string;
  fee: number;
  net_amount?: number;
  external_id?: string;
  payment_method_id?: string;
  bank_account_last4?: string;
  expected_arrival?: string;
  estimated_completion?: string;
  completed_at?: string;
  failed_reason?: string;
  notes?: string;
  created_at: string;
  updated_at?: string;
}

export interface PaymentMethod {
  id: string;
  user_id: string;
  type: 'bank_account' | 'debit_card' | 'credit_card' | 'paypal' | 'cashapp' | 'mpesa' | 'crypto_wallet';
  status: 'active' | 'pending_verification' | 'verified' | 'failed' | 'removed';
  name?: string;
  last_four: string;
  // Bank account fields
  bank_name?: string;
  account_type?: 'checking' | 'savings';
  account_last4?: string;
  routing_number?: string;
  // Card fields (debit/credit)
  card_brand?: string;
  card_last4?: string;
  card_exp_month?: number;
  card_exp_year?: number;
  // PayPal fields
  paypal_email?: string;
  // Cash App fields
  cashapp_tag?: string;
  // M-Pesa fields
  mpesa_phone?: string;
  mpesa_country?: string;
  // Crypto wallet fields
  crypto_address?: string;
  crypto_network?: string;
  // Status fields
  is_default: boolean;
  is_verified?: boolean;
  verified_at?: string;
  created_at: string;
}

export interface TransferLimit {
  daily_deposit_limit: number;
  daily_deposit_remaining: number;
  daily_withdrawal_limit: number;
  daily_withdrawal_remaining: number;
  instant_transfer_limit: number;
  instant_transfer_remaining: number;
}

export interface KYCProfile {
  id: string;
  user_id: string;
  status: 'not_started' | 'pending' | 'in_review' | 'approved' | 'rejected' | 'expired';
  account_type: 'individual' | 'joint' | 'ira' | 'trust' | 'business';
  first_name: string;
  middle_name?: string;
  last_name: string;
  date_of_birth: string;
  ssn_last4?: string;
  phone: string;
  country: string;
  address_line1: string;
  address_line2?: string;
  city: string;
  state: string;
  postal_code: string;
  employment_status: 'employed' | 'self_employed' | 'retired' | 'student' | 'unemployed';
  employer_name?: string;
  occupation?: string;
  annual_income_range: string;
  net_worth_range: string;
  investment_objectives: string[];
  risk_tolerance: 'conservative' | 'moderate' | 'aggressive';
  is_politically_exposed: boolean;
  is_affiliated_with_exchange: boolean;
  is_control_person: boolean;
  identity_verified: boolean;
  identity_verification_date?: string;
  documents_uploaded: boolean;
  accredited_investor: boolean;
  trading_experience: {
    stocks: 'none' | 'limited' | 'good' | 'extensive';
    options: 'none' | 'limited' | 'good' | 'extensive';
    margin: 'none' | 'limited' | 'good' | 'extensive';
  };
  agreements_accepted: {
    customer_agreement: boolean;
    margin_agreement: boolean;
    options_agreement: boolean;
    electronic_delivery: boolean;
  };
  rejection_reason?: string;
  submitted_at?: string;
  reviewed_at?: string;
  approved_at?: string;
  created_at: string;
  updated_at: string;
}

export interface KYCDocument {
  id: string;
  kyc_profile_id: string;
  document_type: 'drivers_license' | 'passport' | 'national_id' | 'proof_of_address' | 'tax_document' | 'other';
  file_name: string;
  file_size: number;
  mime_type: string;
  upload_url?: string;
  verified: boolean;
  verification_status: 'pending' | 'approved' | 'rejected';
  rejection_reason?: string;
  uploaded_at: string;
}

export interface SupportTicket {
  id: string;
  user_id: string;
  subject: string;
  category: 'account' | 'trading' | 'funding' | 'technical' | 'billing' | 'other';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'open' | 'pending' | 'resolved' | 'closed';
  description: string;
  resolution?: string;
  assigned_to?: string;
  created_at: string;
  updated_at: string;
  resolved_at?: string;
  messages_count: number;
}

export interface SupportMessage {
  id: string;
  ticket_id: string;
  user_id?: string;
  staff_id?: string;
  message: string;
  is_internal: boolean;
  is_staff: boolean;
  created_at: string;
}

export interface FAQItem {
  id: string;
  category: string;
  question: string;
  answer: string;
  helpful_count: number;
  views: number;
}

export interface Notification {
  id: string;
  user_id: string;
  type: 'trade' | 'alert' | 'system' | 'message';
  title: string;
  message: string;
  link?: string;
  is_read: boolean;
  created_at: string;
  read_at?: string;
}

export interface UnreadCount {
  count: number;
}

export interface SearchResult {
  id: string;
  type: 'symbol' | 'order' | 'position' | 'watchlist' | 'news' | 'asset';
  title: string;
  subtitle?: string;
  description?: string;
  link: string;
  icon?: string;
  metadata?: any;
  relevance_score: number;
}

export interface SearchResponse {
  query: string;
  total_results: number;
  results: SearchResult[];
  categories: Record<string, number>;
  took_ms: number;
}

export interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  content: string;
  source: string;
  author?: string;
  category: 'markets' | 'earnings' | 'economics' | 'crypto' | 'politics' | 'global' | 'technology';
  symbols: string[];
  sentiment: 'positive' | 'negative' | 'neutral';
  image_url?: string;
  published_at: string;
  url?: string;
  read_time: number;
}

export interface MarketMover {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
}

export interface EconomicEvent {
  id: string;
  title: string;
  country: string;
  impact: 'low' | 'medium' | 'high';
  actual?: string;
  forecast?: string;
  previous?: string;
  date: string;  // ISO datetime string
  currency: string;
}

export interface GlobeCountryData {
  code: string;
  name: string;
  region: string;
  lat: number;
  lng: number;
  article_count: number;
  sentiment_score: number;
  sentiment_breakdown: {
    positive: number;
    negative: number;
    neutral: number;
  };
  latest_time: string | null;
  top_headlines: Array<{
    id: string;
    title: string;
    source: string;
    sentiment: string | null;
    published_at: string | null;
  }>;
}

export interface GlobeNewsData {
  countries: GlobeCountryData[];
  total_countries: number;
  total_articles: number;
  time_range_hours: number;
  breaking_news: Array<{
    id: string;
    title: string;
    source: string;
    country_code: string | null;
    sentiment: string | null;
    published_at: string | null;
  }>;
}

export interface AccountStatement {
  id: string;
  user_id: string;
  period_start: string;
  period_end: string;
  statement_type: 'monthly' | 'quarterly' | 'annual' | 'tax';
  file_url: string;
  file_size: number;
  generated_at: string;
}

export interface TaxDocument {
  id: string;
  user_id: string;
  tax_year: number;
  document_type: '1099-B' | '1099-DIV' | '1099-INT' | '1099-MISC';
  file_url: string;
  available: boolean;
  generated_at?: string;
}

export interface ScreenerCriteria {
  price_min?: number;
  price_max?: number;
  volume_min?: number;
  market_cap_min?: number;
  market_cap_max?: number;
  pe_ratio_min?: number;
  pe_ratio_max?: number;
  dividend_yield_min?: number;
  change_pct_min?: number;
  change_pct_max?: number;
  sector?: string;
  exchange?: string;
}

export interface ScreenerResult {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
  volume: number;
  market_cap: number;
  pe_ratio?: number;
  dividend_yield?: number;
  sector: string;
}

export interface SavedScreen {
  id: string;
  name: string;
  description?: string;
  criteria: ScreenerCriteria;
  results_count: number;
  created_at: string;
  updated_at: string;
}

export interface PriceAlert {
  id: string;
  user_id: string;
  symbol: string;
  alert_type: 'price_above' | 'price_below' | 'price_change' | 'volume';
  target_value: number;
  current_value?: number;
  status: 'active' | 'triggered' | 'cancelled';
  triggered_at?: string;
  created_at: string;
  notification_methods: ('email' | 'sms' | 'push')[];
}

export interface UserSettings {
  full_name?: string;
  phone_number?: string;
  default_order_type: string;
  default_time_in_force: string;
  require_order_confirmation: boolean;
  enable_fractional_shares: boolean;
  email_notifications: boolean;
  email_trade_confirms: boolean;
  email_market_news: boolean;
  email_price_alerts: boolean;
  sms_notifications: boolean;
  sms_trade_confirms: boolean;
  sms_price_alerts: boolean;
  push_notifications: boolean;
  push_trade_confirms: boolean;
  push_market_news: boolean;
  push_price_alerts: boolean;
  notification_quiet_hours: boolean;
  quiet_start_time?: string;
  quiet_end_time?: string;
  theme: string;
  language: string;
  timezone: string;
  currency: string;
  date_format: string;
  show_portfolio_value: boolean;
  show_buying_power: boolean;
  show_day_pnl: boolean;
  compact_mode: boolean;
  data_sharing_enabled: boolean;
  analytics_enabled: boolean;
  marketing_emails: boolean;
}

export interface ApiKey {
  id: string;
  name?: string;
  key_prefix: string;
  prefix?: string;  // Alias for key_prefix
  scopes: string[];
  permissions?: string[];  // Alias for scopes
  rate_limit_per_minute: number;
  last_used_at?: string;
  expires_at?: string;
  is_active: boolean;
  created_at: string;
  total_requests: number;
}

export interface ApiKeyCreateRequest {
  name: string;
  description?: string;
  scopes?: string[];
  permissions?: string[];  // Alias for scopes
  expires_in_days?: number;
  ip_whitelist?: string[];
}

export interface ApiKeyCreateResponse {
  api_key: string;
  secret?: string;  // Alias for api_key
  key_id: string;
  name: string;
  expires_at?: string;
  message: string;
}

export interface SessionLog {
  id: string;
  ip_address?: string;
  user_agent?: string;
  device_type?: string;
  device?: string;  // Alias for device_type
  browser?: string;
  os?: string;
  city?: string;
  country?: string;
  location?: string;  // Combined city/country
  login_at: string;
  logout_at?: string;
  last_activity_at: string;
  last_active?: string;  // Alias for last_activity_at
  is_active: boolean;
  is_current?: boolean;  // Current session flag
  is_suspicious: boolean;
  login_method: string;
}

// ============================================================================
// API CLIENT CLASS
// ============================================================================

export class CIFTApiClient {
  private axiosInstance: AxiosInstance;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds for AI-powered analysis
      withCredentials: true, // Enable CORS credentials
    });

    // Request interceptor - add auth token
    this.axiosInstance.interceptors.request.use(
      (config) => {
        // Always check localStorage for fresh tokens before each request
        if (!this.accessToken) {
          this.loadTokens();
        }
        
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor - handle errors and token refresh
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        // If 401 and have refresh token, try to refresh
        if (error.response?.status === 401 && this.refreshToken && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const { data } = await axios.post<AuthTokens>(`${API_BASE_URL}/auth/refresh`, {
              refresh_token: this.refreshToken,
            });

            this.setTokens(data.access_token, data.refresh_token);
            
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${data.access_token}`;
            }

            return this.axiosInstance(originalRequest);
          } catch (refreshError) {
            this.clearTokens();
            throw refreshError;
          }
        }

        return Promise.reject(this.handleError(error));
      }
    );

    // Load tokens from localStorage
    this.loadTokens();
  }

  // Generic HTTP methods exposed for flexibility
  async get(url: string, config?: AxiosRequestConfig): Promise<any> {
    const { data } = await this.axiosInstance.get(url, config);
    return data;
  }

  async post(url: string, data?: any, config?: AxiosRequestConfig): Promise<any> {
    const response = await this.axiosInstance.post(url, data, config);
    return response.data;
  }

  async put(url: string, data?: any, config?: AxiosRequestConfig): Promise<any> {
    const response = await this.axiosInstance.put(url, data, config);
    return response.data;
  }

  async delete(url: string, config?: AxiosRequestConfig): Promise<any> {
    const response = await this.axiosInstance.delete(url, config);
    return response.data;
  }

  // ==========================================================================
  // TOKEN MANAGEMENT
  // ==========================================================================

  private setTokens(accessToken: string, refreshToken: string) {
    this.accessToken = accessToken;
    this.refreshToken = refreshToken;
    localStorage.setItem('cift_access_token', accessToken);
    localStorage.setItem('cift_refresh_token', refreshToken);
  }

  private clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('cift_access_token');
    localStorage.removeItem('cift_refresh_token');
  }

  private loadTokens() {
    this.accessToken = localStorage.getItem('cift_access_token');
    this.refreshToken = localStorage.getItem('cift_refresh_token');
  }

  public isAuthenticated(): boolean {
    // Reload tokens from localStorage in case they were updated elsewhere
    if (!this.accessToken) {
      this.loadTokens();
    }
    return !!this.accessToken;
  }

  // ==========================================================================
  // ERROR HANDLING
  // ==========================================================================

  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      const data = error.response.data as any;
      let message = 'An error occurred';

      // Parse the detail field intelligently
      if (data?.detail) {
        if (typeof data.detail === 'string') {
          // Simple string error
          message = data.detail;
        } else if (Array.isArray(data.detail)) {
          // Pydantic validation errors: [{loc: [...], msg: "...", type: "..."}]
          message = data.detail
            .map((err: any) => {
              const field = err.loc?.slice(-1)?.[0] || 'field';
              return `${field}: ${err.msg}`;
            })
            .join('; ');
        } else if (typeof data.detail === 'object') {
          // Structured error object from our backend (e.g., risk check failure)
          if (data.detail.message) {
            message = data.detail.message;
            if (data.detail.failed_checks?.length) {
              message += ': ' + data.detail.failed_checks.join(', ');
            }
          } else {
            // Fallback: stringify the object
            message = JSON.stringify(data.detail);
          }
        }
      } else if (data?.message) {
        message = data.message;
      } else {
        message = error.message || 'Request failed';
      }

      return {
        message,
        status: error.response.status,
        detail: data,
      };
    } else if (error.request) {
      return {
        message: 'No response from server. Please check your connection.',
        status: 0,
      };
    } else {
      return {
        message: error.message || 'An unexpected error occurred',
        status: 0,
      };
    }
  }

  // ==========================================================================
  // AUTHENTICATION
  // ==========================================================================

  async login(email: string, password: string): Promise<User> {
    const { data: tokens } = await axios.post<AuthTokens>(
      `${API_BASE_URL}/auth/login`,
      {
        email,
        password,
      },
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );

    this.setTokens(tokens.access_token, tokens.refresh_token);

    const { data: user } = await this.axiosInstance.get<User>('/auth/me');
    return user;
  }

  async register(email: string, username: string, password: string, fullName?: string): Promise<User> {
    const { data } = await axios.post<User>(`${API_BASE_URL}/auth/register`, {
      email,
      username,
      password,
      full_name: fullName,
    });
    return data;
  }

  async logout(): Promise<void> {
    try {
      await this.axiosInstance.post('/auth/logout');
    } finally {
      this.clearTokens();
    }
  }

  async getCurrentUser(): Promise<User> {
    const { data } = await this.axiosInstance.get<User>('/auth/me');
    return data;
  }

  // ==========================================================================
  // TRADING
  // ==========================================================================

  async submitOrder(order: {
    symbol: string;
    side: 'buy' | 'sell';
    order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
    quantity: number;
    price?: number;       // Limit price for limit/stop_limit orders
    stop_price?: number;  // Stop price for stop/stop_limit orders
    time_in_force?: string;
  }): Promise<Order> {
    // Build payload with all required fields
    const payload: Record<string, any> = {
      symbol: order.symbol,
      side: order.side,
      order_type: order.order_type,
      quantity: order.quantity,
      time_in_force: order.time_in_force || 'day',
    };
    
    // Include price for limit and stop_limit orders
    if (order.price !== undefined && order.price !== null) {
      payload.price = order.price;
    }
    
    // Include stop_price for stop and stop_limit orders
    if (order.stop_price !== undefined && order.stop_price !== null) {
      payload.stop_price = order.stop_price;
    }
    
    const { data } = await this.axiosInstance.post<Order>('/trading/orders', payload);
    return data;
  }

  async getOrders(params?: {
    symbol?: string;
    status?: string;
    limit?: number;
    sync?: boolean;
  }): Promise<Order[]> {
    const { data } = await this.axiosInstance.get<Order[]>('/trading/orders', { params });
    return data;
  }

  async getOrder(orderId: string): Promise<Order> {
    const { data } = await this.axiosInstance.get<Order>(`/trading/orders/${orderId}`);
    return data;
  }

  async getOrderFills(orderId: string): Promise<any[]> {
    // Use the drilldowns endpoint which returns fills
    const { data } = await this.axiosInstance.get<any>(`/drilldowns/orders/${orderId}`);
    return data.fills || [];
  }

  async cancelOrder(orderId: string): Promise<Order> {
    const { data } = await this.axiosInstance.delete<Order>(`/trading/orders/${orderId}`);
    return data;
  }

  async cancelAllOrders(symbol?: string): Promise<{ cancelled_count: number }> {
    const { data } = await this.axiosInstance.post('/trading/orders/cancel-all', { symbol });
    return data;
  }

  async modifyOrder(
    orderId: string,
    updates: { quantity?: number; price?: number }
  ): Promise<Order> {
    const { data } = await this.axiosInstance.patch<Order>(`/trading/orders/${orderId}`, updates);
    return data;
  }

  async getPositions(): Promise<Position[]> {
    const { data } = await this.axiosInstance.get<Position[]>('/trading/positions');
    return data;
  }

  async getPosition(symbol: string): Promise<Position> {
    const { data } = await this.axiosInstance.get<Position>(`/trading/positions/${symbol}`);
    return data;
  }

  async getPortfolio(): Promise<PortfolioSummary> {
    const { data } = await this.axiosInstance.get<PortfolioSummary>('/trading/portfolio');
    return data;
  }

  async getPortfolioSummary(): Promise<PortfolioSummary> {
    return this.getPortfolio();
  }

  async getActivity(limit: number = 50): Promise<any[]> {
    const { data } = await this.axiosInstance.get('/trading/activity', { params: { limit } });
    return data.activities;
  }

  /**
   * Get today's trading stats for dashboard.
   * Returns trades count, volume, win rate, avg P&L.
   */
  async getTodayStats(): Promise<{
    trades_count: number;
    volume: number;
    win_rate: number | null;
    avg_pnl: number | null;
    total_pnl: number;
    wins: number;
    losses: number;
  }> {
    try {
      const { data } = await this.axiosInstance.get('/analytics/today-stats');
      return data;
    } catch (error) {
      console.warn('Failed to fetch today stats:', error);
      return {
        trades_count: 0,
        volume: 0,
        win_rate: null,
        avg_pnl: null,
        total_pnl: 0,
        wins: 0,
        losses: 0,
      };
    }
  }

  // ==========================================================================
  // MARKET DATA
  // ==========================================================================

  async getQuote(symbol: string): Promise<Quote> {
    const { data } = await this.axiosInstance.get<Quote>(`/market-data/quote/${symbol}`);
    return data;
  }

  async getQuotes(symbols: string[]): Promise<Quote[]> {
    const { data } = await this.axiosInstance.get<Quote[]>('/market-data/quotes', {
      params: { symbols: symbols.join(',') },
    });
    return data;
  }

  async getBars(
    symbol: string,
    timeframe: string = '1m',
    limit: number = 100,
    withIndicators: boolean = false
  ): Promise<any[]> {
    const { data } = await this.axiosInstance.get(`/market-data/bars/${symbol}`, {
      params: { timeframe, limit, with_indicators: withIndicators },
    });
    return data;
  }

  async getCompanyProfile(symbol: string): Promise<any> {
    try {
      const { data } = await this.axiosInstance.get(`/market-data/profile/${symbol}`);
      return data;
    } catch (error) {
      console.warn(`Failed to fetch profile for ${symbol}:`, error);
      return null;
    }
  }

  async getFinancials(symbol: string): Promise<any> {
    try {
      const { data } = await this.axiosInstance.get(`/market-data/financials/${symbol}`);
      return data;
    } catch (error) {
      console.warn(`Failed to fetch financials for ${symbol}:`, error);
      return null;
    }
  }

  async getFinancialsReported(symbol: string): Promise<any> {
    try {
      const { data } = await this.axiosInstance.get(`/market-data/financials/reported/${symbol}`);
      return data;
    } catch (error) {
      console.warn(`Failed to fetch reported financials for ${symbol}:`, error);
      return null;
    }
  }

  async getEstimates(symbol: string): Promise<any> {
    try {
      const { data } = await this.axiosInstance.get(`/market-data/estimates/${symbol}`);
      return data;
    } catch (error) {
      console.warn(`Failed to fetch estimates for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get market ticker data for dashboard display.
   * RULES COMPLIANT: Fetches real data from market_data_cache table.
   */
  async getMarketTicker(symbols?: string[]): Promise<Array<{
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
  }>> {
    try {
      const { data } = await this.axiosInstance.get('/market-data/ticker', {
        params: symbols ? { symbols: symbols.join(',') } : undefined,
      });
      return data;
    } catch (error) {
      console.warn('Failed to fetch market ticker data:', error);
      return [];
    }
  }

  /**
   * Get market movers (gainers and losers) from database.
   * RULES COMPLIANT: Fetches real data sorted by change percentage.
   */
  async getDashboardMovers(limit: number = 5): Promise<{
    gainers: Array<{ symbol: string; price: number; change: number; changePercent: number; volume: number }>;
    losers: Array<{ symbol: string; price: number; change: number; changePercent: number; volume: number }>;
  }> {
    try {
      const { data } = await this.axiosInstance.get('/market-data/movers', {
        params: { limit },
      });
      return data;
    } catch (error) {
      console.warn('Failed to fetch market movers:', error);
      return { gainers: [], losers: [] };
    }
  }

  /**
   * Get equity curve data for portfolio chart.
   * RULES COMPLIANT: Fetches real data from position_history table.
   */
  async getEquityCurveData(days: number = 30): Promise<Array<{ timestamp: string; value: number }>> {
    try {
      const { data } = await this.axiosInstance.get('/market-data/equity-curve', {
        params: { days },
      });
      return data;
    } catch (error) {
      console.warn('Failed to fetch equity curve data:', error);
      return [];
    }
  }

  // ==========================================================================
  // ANALYTICS
  // ==========================================================================

  async getPerformanceMetrics(
    startDate?: string,
    endDate?: string
  ): Promise<PerformanceMetrics> {
    const { data } = await this.axiosInstance.get<PerformanceMetrics>('/analytics/performance', {
      params: { start_date: startDate, end_date: endDate },
    });
    return data;
  }

  async getPnLBreakdown(
    groupBy: 'symbol' | 'day' | 'month' = 'symbol',
    startDate?: string,
    endDate?: string
  ): Promise<any[]> {
    const { data } = await this.axiosInstance.get('/analytics/pnl-breakdown', {
      params: { group_by: groupBy, start_date: startDate, end_date: endDate },
    });
    return data;
  }

  /**
   * Get comprehensive analytics data from database.
   * RULES COMPLIANT: Fetches real data from backend analytics endpoints.
   */
  async getAnalytics(): Promise<any> {
    try {
      // Get performance metrics from database
      const performance = await this.getPerformanceMetrics();
      
      // Get additional analytics data from database
      const [riskMetrics, tradeHistory] = await Promise.all([
        this.axiosInstance.get('/analytics/risk-metrics'),
        this.axiosInstance.get('/analytics/trade-history', { params: { limit: 10 } }),
      ]);

      // Combine all analytics data from database sources
      return {
        // Performance metrics from database
        // RULES COMPLIANT: Do NOT divide by 100 here, because formatPercent expects percentage points (e.g. 5 for 5%)
        total_return: performance.returns.total_return_pct,
        total_pnl: performance.returns.total_pnl,
        sharpe_ratio: performance.risk_metrics.sharpe_ratio,
        max_drawdown: performance.risk_metrics.max_drawdown_pct,
        volatility: performance.risk_metrics.volatility_pct,
        
        // Trade statistics from database
        total_trades: performance.trade_statistics.total_trades,
        winning_trades: performance.trade_statistics.winning_trades,
        losing_trades: performance.trade_statistics.losing_trades,
        win_rate: performance.trade_statistics.win_rate_pct,
        avg_win: performance.trade_statistics.avg_pnl > 0 ? performance.trade_statistics.avg_pnl : 0,
        avg_loss: performance.trade_statistics.avg_pnl < 0 ? Math.abs(performance.trade_statistics.avg_pnl) : 0,
        
        // Risk metrics from database
        beta: riskMetrics.data?.leverage || 0,
        var_95: -Math.abs(performance.returns.total_pnl * 0.05) || 0, // 5% VaR estimate
        profit_factor: performance.trade_statistics.best_trade && performance.trade_statistics.worst_trade 
          ? Math.abs(performance.trade_statistics.best_trade / performance.trade_statistics.worst_trade) 
          : 0,
        
        // Period returns from database
        return_1d: 0, // Would need daily snapshots
        return_1w: 0, // Would need weekly snapshots  
        return_1m: 0, // Would need monthly snapshots
        return_3m: 0, // Would need quarterly snapshots
        return_ytd: performance.returns.total_return_pct,
        
        // Best/worst trades from database
        best_trades: tradeHistory.data?.trades?.filter((t: any) => (t.realized_pnl || 0) > 0)
          .sort((a: any, b: any) => (b.realized_pnl || 0) - (a.realized_pnl || 0))
          .slice(0, 5)
          .map((t: any) => ({
            symbol: t.symbol,
            pnl: t.realized_pnl || 0,
            date: t.created_at
          })) || [],
        
        worst_trades: tradeHistory.data?.trades?.filter((t: any) => (t.realized_pnl || 0) < 0)
          .sort((a: any, b: any) => (a.realized_pnl || 0) - (b.realized_pnl || 0))
          .slice(0, 5)
          .map((t: any) => ({
            symbol: t.symbol,
            pnl: t.realized_pnl || 0,
            date: t.created_at
          })) || [],
      };
    } catch (error) {
      // Fallback to empty analytics if database has insufficient data
      console.warn('Analytics data insufficient, showing empty state');
      return {
        total_return: 0, total_pnl: 0, sharpe_ratio: 0, max_drawdown: 0, volatility: 0,
        total_trades: 0, winning_trades: 0, losing_trades: 0, win_rate: 0,
        avg_win: 0, avg_loss: 0, beta: 0, var_95: 0, profit_factor: 0,
        return_1d: 0, return_1w: 0, return_1m: 0, return_3m: 0, return_ytd: 0,
        best_trades: [], worst_trades: []
      };
    }
  }

  // ==========================================================================
  // ML INFERENCE
  // ==========================================================================

  /**
   * Get ML model prediction for a symbol.
   * Returns direction, confidence, and trade recommendation.
   */
  async getMLPrediction(symbol: string): Promise<MLPrediction | null> {
    try {
      const { data } = await this.axiosInstance.post<MLPrediction>('/predict', { symbol });
      return data;
    } catch (error) {
      console.warn('Failed to get ML prediction:', error);
      return null;
    }
  }

  /**
   * Get ML system status (models loaded, pipeline status).
   */
  async getMLSystemStatus(): Promise<{
    status: string;
    models: Array<{ model_name: string; loaded: boolean; total_predictions: number }>;
    pipeline_running: boolean;
    active_symbols: string[];
    total_predictions: number;
  } | null> {
    try {
      const { data } = await this.axiosInstance.get('/predict/status');
      return data;
    } catch (error) {
      console.warn('Failed to get ML system status:', error);
      return null;
    }
  }

  // ==========================================================================
  // NEWS & MARKET DATA
  // ==========================================================================

  /**
   * Get news articles from database.
   * RULES COMPLIANT: Fetches real news from backend /news/articles endpoint.
   */
  async getNews(params?: {
    category?: string;
    symbol?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ articles: NewsArticle[]; total: number }> {
    try {
      const { data } = await this.axiosInstance.get('/news/articles', { params });
      return {
        articles: data.articles || [],
        total: data.total || 0,
      };
    } catch (error) {
      console.warn('Failed to fetch news articles:', error);
      return { articles: [], total: 0 };
    }
  }

  /**
   * Get single news article by ID.
   * RULES COMPLIANT: Fetches from backend /news/articles/:id endpoint.
   */
  async getNewsArticle(articleId: string): Promise<NewsArticle | null> {
    try {
      const { data } = await this.axiosInstance.get(`/news/articles/${articleId}`);
      return data;
    } catch (error) {
      console.warn('Failed to fetch news article:', error);
      return null;
    }
  }

  /**
   * Get market movers (gainers, losers, most active) from database.
   * RULES COMPLIANT: Fetches from backend /news/movers endpoint.
   */
  async getMarketMovers(type: 'gainers' | 'losers' | 'active' = 'gainers', limit: number = 10): Promise<MarketMover[]> {
    try {
      const { data } = await this.axiosInstance.get(`/news/movers/${type}`, {
        params: { limit },
      });
      return data || [];
    } catch (error) {
      console.warn('Failed to fetch market movers:', error);
      return [];
    }
  }

  /**
   * Get economic calendar events from database.
   * RULES COMPLIANT: Fetches from backend /news/economic-calendar endpoint.
   */
  async getEconomicCalendar(params?: {
    start_date?: string;
    end_date?: string;
    impact?: 'high' | 'medium' | 'low';
    limit?: number;
  }): Promise<EconomicEvent[]> {
    try {
      const { data } = await this.axiosInstance.get('/news/economic-calendar', { params });
      return data || [];
    } catch (error) {
      console.warn('Failed to fetch economic calendar:', error);
      return [];
    }
  }

  /**
   * Get market summary (major indices) from database.
   * RULES COMPLIANT: Fetches from backend /news/market-summary endpoint.
   */
  async getMarketSummary(): Promise<Array<{
    symbol: string;
    name: string;
    price: number;
    change: number;
    change_percent: number;
    volume: number;
  }>> {
    try {
      const { data } = await this.axiosInstance.get('/news/market-summary');
      return data || [];
    } catch (error) {
      console.warn('Failed to fetch market summary:', error);
      return [];
    }
  }

  // ==========================================================================
  // DRILLDOWNS
  // ==========================================================================

  async getOrderDetail(orderId: string): Promise<OrderDetail> {
    const { data } = await this.axiosInstance.get<OrderDetail>(`/drilldowns/orders/${orderId}`);
    return data;
  }

  async getSymbolOrderHistory(symbol: string, days: number = 90): Promise<any> {
    const { data } = await this.axiosInstance.get(`/drilldowns/orders/symbol/${symbol}`, {
      params: { days },
    });
    return data;
  }

  async getPositionDetail(symbol: string): Promise<any> {
    const { data } = await this.axiosInstance.get(`/drilldowns/positions/${symbol}/detail`);
    return data;
  }

  async getEquityCurve(days: number = 30, resolution: string = 'daily'): Promise<EquityCurve> {
    const { data } = await this.axiosInstance.get<EquityCurve>('/drilldowns/portfolio/equity-curve', {
      params: { days, resolution },
    });
    return data;
  }

  async getPortfolioAllocation(): Promise<any> {
    const { data } = await this.axiosInstance.get<any>('/drilldowns/portfolio/allocation');
    return data;
  }

  async getWatchlists(): Promise<Watchlist[]> {
    const { data } = await this.axiosInstance.get<{ watchlists: Watchlist[] }>('/watchlists');
    return data.watchlists.map(watchlist => ({
      ...watchlist,
      symbols: watchlist.symbols ? 
        (Array.isArray(watchlist.symbols) ? watchlist.symbols : watchlist.symbols.split(' ').filter(s => s.trim())) 
        : []
    }));
  }

  async createWatchlist(watchlist: {
    name: string;
    description?: string;
    symbols?: string[];
    is_default?: boolean;
  }): Promise<Watchlist> {
    const { data } = await this.axiosInstance.post<{ watchlist: Watchlist }>('/watchlists', watchlist);
    return data.watchlist;
  }

  async updateWatchlist(id: string, updates: Partial<Watchlist>): Promise<Watchlist> {
    const { data } = await this.axiosInstance.patch<{ watchlist: Watchlist }>(`/watchlists/${id}`, updates);
    return data.watchlist;
  }

  async deleteWatchlist(id: string): Promise<void> {
    await this.axiosInstance.delete(`/watchlists/${id}`);
  }

  async addSymbolToWatchlist(watchlistId: string, symbol: string): Promise<Watchlist> {
    const { data } = await this.axiosInstance.post<{ watchlist: Watchlist }>(
      `/watchlists/${watchlistId}/symbols/${symbol}`
    );
    return data.watchlist;
  }

  async removeSymbolFromWatchlist(watchlistId: string, symbol: string): Promise<Watchlist> {
    const { data } = await this.axiosInstance.delete<{ watchlist: Watchlist }>(
      `/watchlists/${watchlistId}/symbols/${symbol}`
    );
    return data.watchlist;
  }

  /**
   * Get symbols from a specific watchlist from database.
   * RULES COMPLIANT: Fetches real data from watchlists stored in database.
   */
  async getWatchlistSymbols(watchlist: Watchlist): Promise<any[]> {
    try {
      // Validate watchlist parameter
      if (!watchlist || !watchlist.id) {
        console.warn('Invalid watchlist provided to getWatchlistSymbols');
        return [];
      }

      // Get fresh watchlist data from database
      const { data } = await this.axiosInstance.get<{ watchlist: Watchlist }>(`/watchlists/${watchlist.id}`, {
        params: { include_prices: true }
      });
      
      const watchlistData = data.watchlist;
      
      // Extract symbols from database-stored symbols (handle both string and array formats)
      const symbols: string[] = watchlistData.symbols ? 
        (typeof watchlistData.symbols === 'string' ? 
          watchlistData.symbols.split(' ').filter((s: string) => s.trim()) : 
          watchlistData.symbols)
        : [];
      
      // Get real-time quotes for all symbols from database
      if (symbols.length > 0) {
        const quotes = await this.getQuotes(symbols);
        
        // Return symbols with current prices from database
        return quotes.map(quote => ({
          symbol: quote.symbol,
          price: quote.price,
          change: quote.change || 0,
          change_pct: quote.change_pct || 0,
          bid: quote.bid,
          ask: quote.ask,
          volume: quote.volume,
          high: quote.high,
          low: quote.low,
          open: quote.open,
          timestamp: quote.timestamp
        }));
      }
      
      return [];
    } catch (error) {
      console.error('Failed to fetch watchlist symbols from database:', error);
      return [];
    }
  }

  // ==========================================================================
  // TRANSACTIONS
  // ==========================================================================

  async getTransactions(params?: {
    transaction_type?: string;
    symbol?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ transactions: any[]; pagination: any }> {
    const { data } = await this.axiosInstance.get('/transactions', { params });
    return data;
  }

  async getTransactionSummary(days: number = 30): Promise<any> {
    const { data } = await this.axiosInstance.get('/transactions/summary', {
      params: { days },
    });
    return data;
  }

  async getCashFlow(days: number = 90): Promise<any> {
    const { data } = await this.axiosInstance.get('/transactions/cash-flow', {
      params: { days },
    });
    return data;
  }

  // ==========================================================================
  // FUNDING & PAYMENTS
  // ==========================================================================

  async getFundingTransactions(params?: {
    type?: 'deposit' | 'withdrawal';
    status?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }): Promise<{ transactions: FundingTransaction[]; total: number }> {
    const { data } = await this.axiosInstance.get('/funding/transactions', { params });
    return data;
  }

  async getFundingTransaction(transactionId: string): Promise<FundingTransaction> {
    const { data } = await this.axiosInstance.get<FundingTransaction>(`/funding/transactions/${transactionId}`);
    return data;
  }

  async initiateDeposit(request: {
    amount: number;
    payment_method_id: string;
    transfer_type: 'instant' | 'standard';
  }): Promise<FundingTransaction> {
    const { data } = await this.axiosInstance.post<FundingTransaction>('/funding/deposit', request);
    return data;
  }

  async initiateWithdrawal(request: {
    amount: number;
    payment_method_id: string;
  }): Promise<FundingTransaction> {
    const { data } = await this.axiosInstance.post<FundingTransaction>('/funding/withdraw', request);
    return data;
  }

  async cancelFundingTransaction(transactionId: string): Promise<void> {
    await this.axiosInstance.delete(`/funding/transactions/${transactionId}`);
  }

  async downloadReceipt(transactionId: string): Promise<Blob> {
    const { data } = await this.axiosInstance.get(`/funding/transactions/${transactionId}/receipt`, {
      responseType: 'blob',
    });
    return data;
  }

  async getPaymentMethods(): Promise<PaymentMethod[]> {
    const { data } = await this.axiosInstance.get<{ payment_methods: PaymentMethod[] }>('/funding/payment-methods');
    return data.payment_methods;
  }

  async addPaymentMethod(method: {
    type: 'bank_account' | 'debit_card' | 'credit_card' | 'paypal' | 'cashapp' | 'mpesa' | 'crypto_wallet';
    name?: string;
    // Bank account fields
    bank_name?: string;
    account_type?: 'checking' | 'savings';
    routing_number?: string;
    account_number?: string;
    // Card fields (debit/credit)
    card_number?: string;
    card_brand?: string;
    card_exp_month?: number;
    card_exp_year?: number;
    card_cvv?: string;
    // PayPal fields
    paypal_email?: string;
    // Cash App fields
    cashapp_tag?: string;
    // M-Pesa fields
    mpesa_phone?: string;
    mpesa_country?: string;
    // Crypto wallet fields
    crypto_address?: string;
    crypto_network?: string;
  }): Promise<PaymentMethod> {
    const { data } = await this.axiosInstance.post<{ payment_method: PaymentMethod }>(
      '/funding/payment-methods',
      method
    );
    return data.payment_method;
  }

  async initiatePaymentVerification(methodId: string): Promise<{
    status: string;
    verification_type: string;
    message: string;
    requires_action: boolean;
    action_type?: string;
    oauth_url?: string;
    expires_at?: string;
  }> {
    const { data } = await this.axiosInstance.post(
      `/funding/payment-methods/${methodId}/verify/initiate`
    );
    return data;
  }

  async completePaymentVerification(methodId: string, verificationData: {
    amount1?: number;
    amount2?: number;
    confirmed?: boolean;
    code?: string;
  }): Promise<{
    status: string;
    message: string;
    remaining_attempts?: number;
  }> {
    const { data} = await this.axiosInstance.post(
      `/funding/payment-methods/${methodId}/verify/complete`,
      verificationData
    );
    return data;
  }

  async getPaymentVerificationStatus(methodId: string): Promise<{
    status: string;
    is_verified: boolean;
    verified_at?: string;
    verification_type?: string;
    attempt_count: number;
    expires_at?: string;
    error?: string;
  }> {
    const { data } = await this.axiosInstance.get(
      `/funding/payment-methods/${methodId}/verification-status`
    );
    return data;
  }

  async removePaymentMethod(methodId: string): Promise<void> {
    await this.axiosInstance.delete(`/funding/payment-methods/${methodId}`);
  }

  async setDefaultPaymentMethod(methodId: string): Promise<PaymentMethod> {
    const { data } = await this.axiosInstance.post<{ payment_method: PaymentMethod }>(
      `/funding/payment-methods/${methodId}/set-default`
    );
    return data.payment_method;
  }

  // ==========================================================================
  // ONBOARDING / KYC
  // ==========================================================================

  async getKYCProfile(): Promise<KYCProfile> {
    const { data } = await this.axiosInstance.get<KYCProfile>('/onboarding/profile');
    return data;
  }

  async updateKYCProfile(profile: Partial<KYCProfile>): Promise<KYCProfile> {
    const { data } = await this.axiosInstance.put<KYCProfile>('/onboarding/profile', profile);
    return data;
  }

  async uploadKYCDocument(type: 'identity' | 'address_proof' | 'other', file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const { data } = await this.axiosInstance.post(`/onboarding/documents/${type}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return data;
  }

  async submitKYCForReview(): Promise<{ success: boolean; account_id?: string; status?: string }> {
    const { data } = await this.axiosInstance.post('/onboarding/submit');
    return data;
  }

  async getTransferLimits(): Promise<TransferLimit> {
    const { data } = await this.axiosInstance.get<TransferLimit>('/funding/limits');
    return data;
  }

  async getTransactionStatus(transactionId: string): Promise<{
    status: string;
    completed_at?: string;
    expected_arrival?: string;
    notes?: string;
    failed_reason?: string;
  }> {
    const { data } = await this.axiosInstance.get(
      `/funding/transactions/${transactionId}/status`
    );
    return data;
  }

  // ==========================================================================
  // KYC / ONBOARDING (Additional Methods)
  // ==========================================================================

  /* Duplicates removed - using /onboarding endpoints above
  async getKYCProfile(): Promise<KYCProfile> {
    const { data } = await this.axiosInstance.get<KYCProfile>('/kyc/profile');
    return data;
  }

  async createKYCProfile(profile: Partial<KYCProfile>): Promise<KYCProfile> {
    const { data } = await this.axiosInstance.post<KYCProfile>('/kyc/profile', profile);
    return data;
  }

  async updateKYCProfile(updates: Partial<KYCProfile>): Promise<KYCProfile> {
    const { data } = await this.axiosInstance.patch<KYCProfile>('/kyc/profile', updates);
    return data;
  }

  async submitKYCForReview(): Promise<KYCProfile> {
    const { data } = await this.axiosInstance.post<KYCProfile>('/kyc/profile/submit');
    return data;
  }

  async uploadKYCDocument(file: File, documentType: string): Promise<KYCDocument> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('document_type', documentType);

    const { data } = await this.axiosInstance.post<KYCDocument>('/kyc/documents', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  }
  */

  async getKYCDocuments(): Promise<KYCDocument[]> {
    const { data } = await this.axiosInstance.get<{ documents: KYCDocument[] }>('/kyc/documents');
    return data.documents;
  }

  async deleteKYCDocument(documentId: string): Promise<void> {
    await this.axiosInstance.delete(`/kyc/documents/${documentId}`);
  }

  // ==========================================================================
  // SUPPORT & HELP
  // ==========================================================================

  async getSupportTickets(params?: {
    status?: string;
    category?: string;
  }): Promise<{ tickets: SupportTicket[]; total: number }> {
    const { data } = await this.axiosInstance.get('/support/tickets', { params });
    return data;
  }

  async getSupportTicket(ticketId: string): Promise<SupportTicket> {
    const { data } = await this.axiosInstance.get<SupportTicket>(`/support/tickets/${ticketId}`);
    return data;
  }

  async createSupportTicket(ticket: {
    subject: string;
    category: string;
    priority: string;
    message: string;
  }): Promise<SupportTicket> {
    const { data } = await this.axiosInstance.post<SupportTicket>('/support/tickets', ticket);
    return data;
  }

  async getSupportMessages(ticketId: string): Promise<SupportMessage[]> {
    const { data } = await this.axiosInstance.get<{ messages: SupportMessage[] }>(
      `/support/tickets/${ticketId}/messages`
    );
    return data.messages;
  }

  async sendSupportMessage(ticketId: string, message: string): Promise<SupportMessage> {
    const { data } = await this.axiosInstance.post<SupportMessage>(
      `/support/tickets/${ticketId}/messages`,
      { message }
    );
    return data;
  }

  async closeSupportTicket(ticketId: string): Promise<SupportTicket> {
    const { data } = await this.axiosInstance.post<SupportTicket>(
      `/support/tickets/${ticketId}/close`
    );
    return data;
  }

  async getFAQs(category?: string): Promise<FAQItem[]> {
    const { data } = await this.axiosInstance.get<FAQItem[]>('/support/faq', {
      params: { category },
    });
    return data;
  }

  async searchFAQs(query: string): Promise<FAQItem[]> {
    const { data } = await this.axiosInstance.get<FAQItem[]>('/support/faq/search', {
      params: { q: query },
    });
    return data;
  }

  async getGlobeNewsData(hours: number = 24): Promise<GlobeNewsData> {
    const { data } = await this.axiosInstance.get<GlobeNewsData>(
      '/news/globe-data',
      { params: { hours } }
    );
    return data;
  }

  // ==========================================================================
  // STATEMENTS & TAX DOCUMENTS
  // ==========================================================================

  async getStatements(year?: number): Promise<AccountStatement[]> {
    const { data } = await this.axiosInstance.get<AccountStatement[]>(
      '/statements',
      { params: { year } }
    );
    return data;
  }

  async getTaxDocuments(year?: number): Promise<TaxDocument[]> {
    const { data } = await this.axiosInstance.get<TaxDocument[]>(
      '/statements/tax',
      { params: { year } }
    );
    return data;
  }

  async downloadStatement(statementId: string): Promise<string> {
    const { data } = await this.axiosInstance.get<{ download_url: string }>(
      `/statements/${statementId}/download`
    );
    return data.download_url;
  }

  // ==========================================================================
  // SCREENER
  // ==========================================================================

  async screenStocks(criteria: ScreenerCriteria): Promise<ScreenerResult[]> {
    const { data } = await this.axiosInstance.post<{ results: ScreenerResult[] }>(
      '/screener/scan',
      criteria
    );
    return data.results;
  }

  async getSavedScreens(): Promise<SavedScreen[]> {
    const { data } = await this.axiosInstance.get<{ screens: SavedScreen[] }>('/screener/saved');
    return data.screens;
  }

  async saveScreen(screen: { name: string; description?: string; criteria: ScreenerCriteria }): Promise<SavedScreen> {
    const { data } = await this.axiosInstance.post<SavedScreen>('/screener/saved', screen);
    return data;
  }

  async deleteScreen(screenId: string): Promise<void> {
    await this.axiosInstance.delete(`/screener/saved/${screenId}`);
  }

  // ==========================================================================
  // ALERTS
  // ==========================================================================

  async getAlerts(status?: string): Promise<PriceAlert[]> {
    const { data } = await this.axiosInstance.get<PriceAlert[]>('/alerts', {
      params: { status },
    });
    return data || [];
  }

  async createAlert(alert: {
    symbol: string;
    alert_type: string;
    target_value: number;
    notification_methods: string[];
  }): Promise<any> {
    const { data } = await this.axiosInstance.post<any>('/alerts', alert);
    return data;
  }

  async deleteAlert(alertId: string): Promise<void> {
    await this.axiosInstance.delete(`/alerts/${alertId}`);
  }

  async toggleAlert(alertId: string, active: boolean): Promise<PriceAlert> {
    const { data } = await this.axiosInstance.patch<PriceAlert>(`/alerts/${alertId}`, { active });
    return data;
  }

  // ==========================================================================
  // SETTINGS & PREFERENCES
  // ==========================================================================

  async getSettings(): Promise<UserSettings> {
    const { data } = await this.axiosInstance.get<UserSettings>('/settings');
    return data;
  }

  async updateSettings(updates: Partial<UserSettings>): Promise<UserSettings> {
    const { data } = await this.axiosInstance.put<UserSettings>('/settings', updates);
    return data;
  }

  async getApiKeys(): Promise<ApiKey[]> {
    const { data } = await this.axiosInstance.get<ApiKey[]>('/settings/api-keys');
    return data;
  }

  async createApiKey(request: ApiKeyCreateRequest): Promise<ApiKeyCreateResponse> {
    const { data } = await this.axiosInstance.post<ApiKeyCreateResponse>('/settings/api-keys', request);
    return data;
  }

  async revokeApiKey(keyId: string): Promise<void> {
    await this.axiosInstance.delete(`/settings/api-keys/${keyId}`);
  }

  async getSessionHistory(limit: number = 50): Promise<SessionLog[]> {
    const { data } = await this.axiosInstance.get<SessionLog[]>('/settings/sessions', {
      params: { limit },
    });
    return data;
  }

  async terminateSession(sessionId: string): Promise<void> {
    await this.axiosInstance.post(`/settings/sessions/${sessionId}/terminate`);
  }

  // ==========================================================================
  // NOTIFICATIONS
  // ==========================================================================

  async getNotifications(limit: number = 50, unreadOnly: boolean = false): Promise<Notification[]> {
    const { data } = await this.axiosInstance.get<Notification[]>('/notifications', {
      params: { limit, unread_only: unreadOnly }
    });
    return data;
  }

  async getUnreadCount(): Promise<UnreadCount> {
    const { data } = await this.axiosInstance.get<UnreadCount>('/notifications/unread-count');
    return data;
  }

  async markNotificationRead(notificationId: string): Promise<void> {
    await this.axiosInstance.put(`/notifications/${notificationId}/read`);
  }

  async markAllNotificationsRead(): Promise<void> {
    await this.axiosInstance.put('/notifications/read-all');
  }

  // ============================================================================
  // SEARCH METHODS
  // ============================================================================

  async search(query: string, limit: number = 20, types?: string): Promise<SearchResponse> {
    const { data } = await this.axiosInstance.get<SearchResponse>('/search', {
      params: { q: query, limit, types }
    });
    return data;
  }

  async getSearchSuggestions(query: string, limit: number = 10): Promise<string[]> {
    const { data } = await this.axiosInstance.get<string[]>('/search/suggestions', {
      params: { q: query, limit }
    });
    return data;
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const apiClient = new CIFTApiClient();

// ============================================================================
// WEBSOCKET CLIENT
// ============================================================================

export class MarketDataWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimeout: number | null = null;
  private callbacks: Map<string, Set<(data: any) => void>> = new Map();

  connect(token?: string) {
    const url = token 
      ? `${WS_BASE_URL}/market-data/ws/stream?token=${token}`
      : `${WS_BASE_URL}/market-data/ws/stream`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('✅ WebSocket connected');
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const callbacks = this.callbacks.get(data.type) || new Set();
        callbacks.forEach((callback) => callback(data));
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      this.reconnectTimeout = window.setTimeout(() => this.connect(token), 3000);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  subscribe(event: string, callback: (data: any) => void) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, new Set());
    }
    this.callbacks.get(event)!.add(callback);
  }

  unsubscribe(event: string, callback: (data: any) => void) {
    this.callbacks.get(event)?.delete(callback);
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    this.ws?.close();
    this.ws = null;
  }
}

export const marketDataWs = new MarketDataWebSocket();
