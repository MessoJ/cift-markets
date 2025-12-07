import { createSignal, onMount, onCleanup, Show } from 'solid-js';
import { CheckCircle, Clock, AlertCircle, Loader } from 'lucide-solid';
import { apiClient } from '../../../lib/api/client';

interface TransactionStatusTrackerProps {
  transactionId: string;
  initialStatus: string;
  onStatusChange?: (status: string) => void;
}

export function TransactionStatusTracker(props: TransactionStatusTrackerProps) {
  const [status, setStatus] = createSignal(props.initialStatus);
  const [expectedArrival, setExpectedArrival] = createSignal<string | null>(null);
  const [failedReason, setFailedReason] = createSignal<string | null>(null);
  const [completedAt, setCompletedAt] = createSignal<string | null>(null);
  
  let pollInterval: number | null = null;

  onMount(() => {
    // Start polling if transaction is not in a final state
    if (shouldPoll(props.initialStatus)) {
      startPolling();
    }
  });

  onCleanup(() => {
    stopPolling();
  });

  const shouldPoll = (currentStatus: string): boolean => {
    // Poll for pending and processing statuses
    return currentStatus === 'pending' || currentStatus === 'processing';
  };

  const startPolling = () => {
    // Poll every 5 seconds
    pollInterval = window.setInterval(async () => {
      await checkStatus();
    }, 5000);
    
    // Also check immediately
    checkStatus();
  };

  const stopPolling = () => {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  };

  const checkStatus = async () => {
    try {
      const result = await apiClient.getTransactionStatus(props.transactionId);
      
      const newStatus = result.status;
      setStatus(newStatus);
      setExpectedArrival(result.expected_arrival || null);
      setCompletedAt(result.completed_at || null);
      setFailedReason(result.failed_reason || null);
      
      // Notify parent of status change
      if (props.onStatusChange && newStatus !== props.initialStatus) {
        props.onStatusChange(newStatus);
      }
      
      // Stop polling if reached a final state
      if (!shouldPoll(newStatus)) {
        stopPolling();
      }
    } catch (err) {
      console.error('Failed to check transaction status:', err);
      // Don't stop polling on error, will retry
    }
  };

  const getStatusIcon = () => {
    switch (status()) {
      case 'completed':
        return <CheckCircle size={14} class="text-success-500" />;
      case 'failed':
      case 'cancelled':
        return <AlertCircle size={14} class="text-danger-500" />;
      case 'processing':
        return <Loader size={14} class="text-info-500 animate-spin" />;
      case 'pending':
      default:
        return <Clock size={14} class="text-warning-500" />;
    }
  };

  const getStatusColor = () => {
    switch (status()) {
      case 'completed':
        return 'text-success-500';
      case 'failed':
      case 'cancelled':
        return 'text-danger-500';
      case 'processing':
        return 'text-info-500';
      case 'pending':
      default:
        return 'text-warning-500';
    }
  };

  const getStatusText = () => {
    const statusText = status().replace(/_/g, ' ');
    return statusText.charAt(0).toUpperCase() + statusText.slice(1);
  };

  const formatDateTime = (dateString: string | null) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div class="space-y-1">
      <div class={`inline-flex items-center gap-1.5 ${getStatusColor()}`}>
        {getStatusIcon()}
        <span class="text-xs font-semibold">{getStatusText()}</span>
      </div>
      
      <Show when={status() === 'pending' || status() === 'processing'}>
        <Show when={expectedArrival()}>
          <div class="text-xs text-gray-500">
            Expected: {formatDateTime(expectedArrival())}
          </div>
        </Show>
      </Show>
      
      <Show when={status() === 'completed' && completedAt()}>
        <div class="text-xs text-gray-500">
          Completed: {formatDateTime(completedAt())}
        </div>
      </Show>
      
      <Show when={status() === 'failed' && failedReason()}>
        <div class="text-xs text-danger-500">
          Reason: {failedReason()}
        </div>
      </Show>
    </div>
  );
}
