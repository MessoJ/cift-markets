/**
 * Authentication Store
 * 
 * Manages user authentication state using SolidJS signals.
 */

import { createSignal } from 'solid-js';
import { apiClient, User } from '~/lib/api/client';

// ============================================================================
// STATE
// ============================================================================

const [user, setUser] = createSignal<User | null>(null);
const [isLoading, setIsLoading] = createSignal(true);
const [isAuthenticated, setIsAuthenticated] = createSignal(false);

// ============================================================================
// ACTIONS
// ============================================================================

async function login(email: string, password: string): Promise<void> {
  setIsLoading(true);
  try {
    const userData = await apiClient.login(email, password);
    setUser(userData);
    setIsAuthenticated(true);
  } catch (error) {
    setIsAuthenticated(false);
    throw error;
  } finally {
    setIsLoading(false);
  }
}

async function register(
  email: string,
  username: string,
  password: string,
  fullName?: string
): Promise<void> {
  setIsLoading(true);
  try {
    await apiClient.register(email, username, password, fullName);
    // After registration, log in
    await login(email, password);
  } catch (error) {
    throw error;
  } finally {
    setIsLoading(false);
  }
}

async function logout(): Promise<void> {
  setIsLoading(true);
  try {
    await apiClient.logout();
  } finally {
    setUser(null);
    setIsAuthenticated(false);
    setIsLoading(false);
  }
}

async function checkAuth(): Promise<void> {
  // Always check if tokens exist in localStorage first
  const hasToken = localStorage.getItem('cift_access_token');
  
  if (!hasToken) {
    console.log('No access token found, user not authenticated');
    setUser(null);
    setIsAuthenticated(false);
    setIsLoading(false);
    return;
  }

  console.log('Access token found, validating with server...');
  
  // Add timeout to prevent infinite loading
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Authentication check timed out after 10 seconds')), 10000);
  });
  
  try {
    // Race between API call and timeout
    const userData = await Promise.race([
      apiClient.getCurrentUser(),
      timeoutPromise
    ]);
    
    console.log('Authentication successful:', userData.username);
    setUser(userData);
    setIsAuthenticated(true);
  } catch (error: any) {
    // If 401 or any auth error, clear everything
    console.warn('Authentication check failed:', error?.response?.status || error?.message);
    
    // If it's a timeout or network error, keep tokens but mark as not authenticated
    // User can try to login again
    if (error?.message?.includes('timeout') || error?.status === 0) {
      console.error('API server appears to be unreachable or slow. Please check backend is running.');
    }
    
    setUser(null);
    setIsAuthenticated(false);
    
    // Clear invalid tokens only if it's an auth error (401)
    if (error?.response?.status === 401) {
      console.log('Clearing invalid tokens due to 401 error');
      localStorage.removeItem('cift_access_token');
      localStorage.removeItem('cift_refresh_token');
    }
  } finally {
    // ALWAYS set loading to false to prevent infinite loading
    setIsLoading(false);
  }
}

// Check auth on initialization
checkAuth();

// ============================================================================
// EXPORTS
// ============================================================================

async function debugAuth(): Promise<void> {
  console.log('=== AUTH DEBUG ===');
  console.log('Tokens in localStorage:');
  console.log('- access_token:', localStorage.getItem('cift_access_token') ? 'EXISTS' : 'MISSING');
  console.log('- refresh_token:', localStorage.getItem('cift_refresh_token') ? 'EXISTS' : 'MISSING');
  console.log('Auth store state:');
  console.log('- isAuthenticated:', isAuthenticated());
  console.log('- isLoading:', isLoading());
  console.log('- user:', user()?.username || 'null');
  console.log('API client state:');
  console.log('- isAuthenticated:', apiClient.isAuthenticated());
  
  // Try a test API call
  try {
    const userData = await apiClient.getCurrentUser();
    console.log('✅ getCurrentUser() works:', userData.username);
  } catch (error: any) {
    console.log('❌ getCurrentUser() failed:', error?.response?.status, error?.message);
  }
  console.log('=== END DEBUG ===');
}

// Make debug function available globally for development
(window as any).debugAuth = debugAuth;

export const authStore = {
  user,
  isLoading,
  isAuthenticated,
  login,
  register,
  logout,
  checkAuth,
  debugAuth,
};
