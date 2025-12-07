/**
 * SUPPORT TICKET DETAIL PAGE
 * View ticket details and communicate with support team
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import {
  ArrowLeft,
  Send,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from 'lucide-solid';
import { apiClient, SupportTicket, SupportMessage } from '../../lib/api/client';

export default function TicketDetailPage() {
  const params = useParams();
  const navigate = useNavigate();
  
  const [ticket, setTicket] = createSignal<SupportTicket | null>(null);
  const [messages, setMessages] = createSignal<SupportMessage[]>([]);
  const [newMessage, setNewMessage] = createSignal('');
  const [loading, setLoading] = createSignal(true);
  const [sending, setSending] = createSignal(false);
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);

  // Auto-hide notification
  createEffect(() => {
    if (notification()) {
      setTimeout(() => setNotification(null), 5000);
    }
  });

  const loadTicket = async () => {
    try {
      setLoading(true);
      const [ticketData, messagesData] = await Promise.all([
        apiClient.getSupportTicket(params.id),
        apiClient.getSupportMessages(params.id),
      ]);
      setTicket(ticketData);
      setMessages(messagesData);
    } catch (err: any) {
      console.error('Failed to load ticket:', err);
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to load ticket'
      });
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    loadTicket();
  });

  const handleSendMessage = async () => {
    if (!newMessage().trim() || newMessage().length < 5) {
      setNotification({type: 'error', message: 'Message must be at least 5 characters'});
      return;
    }

    setSending(true);
    try {
      await apiClient.sendSupportMessage(params.id, newMessage());
      setNotification({type: 'success', message: 'Message sent successfully'});
      setNewMessage('');
      await loadTicket(); // Reload messages
    } catch (err: any) {
      console.error('Failed to send message:', err);
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to send message'
      });
    } finally {
      setSending(false);
    }
  };

  const handleCloseTicket = async () => {
    if (!confirm('Are you sure you want to close this ticket?')) return;

    try {
      await apiClient.closeSupportTicket(params.id);
      setNotification({type: 'success', message: 'Ticket closed successfully'});
      setTimeout(() => navigate('/support'), 1500);
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to close ticket'
      });
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'open': return <Clock size={16} class="text-warning-500" />;
      case 'in_progress': return <AlertCircle size={16} class="text-primary-500" />;
      case 'resolved': return <CheckCircle2 size={16} class="text-success-500" />;
      case 'closed': return <XCircle size={16} class="text-gray-500" />;
      default: return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'text-warning-500 bg-warning-500/10';
      case 'in_progress': return 'text-primary-500 bg-primary-500/10';
      case 'resolved': return 'text-success-500 bg-success-500/10';
      case 'closed': return 'text-gray-500 bg-gray-800/50';
      default: return 'text-gray-400 bg-gray-800/50';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'text-danger-500 bg-danger-500/10';
      case 'high': return 'text-warning-500 bg-warning-500/10';
      case 'medium': return 'text-primary-500 bg-primary-500/10';
      case 'low': return 'text-gray-500 bg-gray-800/50';
      default: return 'text-gray-400 bg-gray-800/50';
    }
  };

  return (
    <div class="h-full flex flex-col gap-2 sm:gap-3 p-2 sm:p-3">
      {/* Inline Notification */}
      <Show when={notification()}>
        <div class={`p-4 rounded-lg border ${
          notification()?.type === 'success' 
            ? 'bg-success-500/10 border-success-500/30 text-success-500' 
            : 'bg-danger-500/10 border-danger-500/30 text-danger-500'
        } flex items-center justify-between`}>
          <div class="flex items-center gap-3">
            {notification()?.type === 'success' ? (
              <CheckCircle2 size={20} />
            ) : (
              <XCircle size={20} />
            )}
            <span class="text-sm font-semibold">{notification()?.message}</span>
          </div>
          <button 
            onClick={() => setNotification(null)}
            class="p-1 hover:bg-white/10 rounded transition-colors"
          >
            <XCircle size={16} />
          </button>
        </div>
      </Show>

      {/* Header */}
      <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
        <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-0">
          <div class="flex items-center gap-4">
            <button
              onClick={() => navigate('/support')}
              class="p-2 hover:bg-terminal-800 rounded transition-colors"
            >
              <ArrowLeft size={20} class="text-gray-400" />
            </button>
            <div>
              <h1 class="text-lg font-bold text-white">Support Ticket</h1>
              <p class="text-xs text-gray-400">Ticket ID: {params.id.slice(0, 8)}</p>
            </div>
          </div>
          <Show when={ticket() && ticket()!.status !== 'closed'}>
            <button
              onClick={handleCloseTicket}
              class="px-4 py-2 bg-terminal-800 hover:bg-terminal-750 text-white text-sm font-semibold rounded transition-colors"
            >
              Close Ticket
            </button>
          </Show>
        </div>
      </div>

      <Show when={loading()}>
        <div class="flex-1 flex items-center justify-center">
          <div class="text-gray-500">Loading ticket...</div>
        </div>
      </Show>

      <Show when={!loading() && ticket()}>
        <div class="flex-1 flex flex-col lg:flex-row gap-2 sm:gap-3 overflow-hidden">
          {/* Main Content - Messages */}
          <div class="flex-1 flex flex-col gap-2 sm:gap-3 overflow-hidden">
            {/* Ticket Info Card */}
            <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
              <h2 class="text-base sm:text-lg font-bold text-white mb-2">{ticket()?.subject}</h2>
              <div class="flex items-center gap-3 text-sm">
                <div class={`flex items-center gap-1.5 px-2 py-1 rounded ${getStatusColor(ticket()!.status)}`}>
                  {getStatusIcon(ticket()!.status)}
                  <span class="font-semibold capitalize">{ticket()?.status?.replace('_', ' ')}</span>
                </div>
                <div class={`flex items-center gap-1.5 px-2 py-1 rounded ${ticket() ? getPriorityColor(ticket()!.priority) : 'text-gray-400 bg-gray-800/50'}`}>
                  <span class="font-semibold capitalize">{ticket()?.priority}</span>
                </div>
                <div class="text-gray-500">
                  Created {new Date(ticket()!.created_at).toLocaleString()}
                </div>
              </div>
            </div>

            {/* Messages */}
            <div class="flex-1 bg-terminal-900 border border-terminal-750 flex flex-col overflow-hidden">
              <div class="p-3 sm:p-4 border-b border-terminal-750">
                <h3 class="text-sm font-bold text-white">Conversation</h3>
              </div>
              
              <div class="flex-1 overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4">
                <For each={messages()}>
                  {(msg) => (
                    <div class={`flex ${msg.is_staff ? 'justify-start' : 'justify-end'}`}>
                      <div class={`max-w-[70%] ${
                        msg.is_staff 
                          ? 'bg-terminal-850 border border-terminal-750' 
                          : 'bg-accent-500/10 border border-accent-500/30'
                      } p-4 rounded-lg`}>
                        <div class="flex items-center gap-2 mb-2">
                          <span class={`text-xs font-semibold ${msg.is_staff ? 'text-accent-500' : 'text-primary-500'}`}>
                            {msg.is_staff ? 'Support Team' : 'You'}
                          </span>
                          <span class="text-xs text-gray-500">
                            {new Date(msg.created_at).toLocaleString()}
                          </span>
                        </div>
                        <p class="text-sm text-gray-300 whitespace-pre-wrap">{msg.message}</p>
                      </div>
                    </div>
                  )}
                </For>
                
                <Show when={messages().length === 0}>
                  <div class="text-center text-gray-500 py-8">
                    No messages yet. Start the conversation below.
                  </div>
                </Show>
              </div>

              {/* Message Input */}
              <Show when={ticket()!.status !== 'closed'}>
                <div class="p-4 border-t border-terminal-750">
                  <div class="flex gap-3">
                    <textarea
                      value={newMessage()}
                      onInput={(e) => setNewMessage(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                      placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
                      rows={3}
                      class="flex-1 bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded-lg focus:outline-none focus:border-accent-500 resize-none"
                    />
                    <button
                      onClick={handleSendMessage}
                      disabled={sending() || !newMessage().trim()}
                      class="px-6 py-3 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors flex items-center gap-2"
                    >
                      <Send size={18} />
                      {sending() ? 'Sending...' : 'Send'}
                    </button>
                  </div>
                  <div class="text-xs text-gray-500 mt-2">
                    Minimum 5 characters. Press Enter to send, Shift+Enter for new line.
                  </div>
                </div>
              </Show>

              <Show when={ticket()!.status === 'closed'}>
                <div class="p-4 border-t border-terminal-750 bg-terminal-850">
                  <div class="text-center text-gray-500 text-sm">
                    This ticket is closed. No further messages can be sent.
                  </div>
                </div>
              </Show>
            </div>
          </div>

          {/* Sidebar - Ticket Details */}
          <div class="lg:w-80 flex flex-col gap-2 sm:gap-3">
            <div class="bg-terminal-900 border border-terminal-750 p-4">
              <h3 class="text-sm font-bold text-white mb-4">Ticket Details</h3>
              <div class="space-y-3 text-sm">
                <div>
                  <div class="text-xs text-gray-500 uppercase mb-1">Category</div>
                  <div class="text-white capitalize">{ticket()?.category?.replace('_', ' ')}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500 uppercase mb-1">Status</div>
                  <div class={`inline-flex items-center gap-1.5 px-2 py-1 rounded ${ticket() ? getStatusColor(ticket()!.status) : ''}`}>
                    {ticket() && getStatusIcon(ticket()!.status)}
                    <span class="font-semibold capitalize">{ticket()?.status?.replace('_', ' ')}</span>
                  </div>
                </div>
                <div>
                  <div class="text-xs text-gray-500 uppercase mb-1">Priority</div>
                  <div class={`inline-flex items-center gap-1.5 px-2 py-1 rounded ${getPriorityColor(ticket()!.priority)}`}>
                    <span class="font-semibold capitalize">{ticket()?.priority}</span>
                  </div>
                </div>
                <div>
                  <div class="text-xs text-gray-500 uppercase mb-1">Created</div>
                  <div class="text-white">{new Date(ticket()!.created_at).toLocaleString()}</div>
                </div>
                <Show when={ticket()!.updated_at}>
                  <div>
                    <div class="text-xs text-gray-500 uppercase mb-1">Last Updated</div>
                    <div class="text-white">{new Date(ticket()!.updated_at!).toLocaleString()}</div>
                  </div>
                </Show>
              </div>
            </div>

            <div class="bg-terminal-900 border border-terminal-750 p-4">
              <h3 class="text-sm font-bold text-white mb-4">Need Help?</h3>
              <div class="space-y-3 text-sm text-gray-400">
                <p>Our support team typically responds within 2-4 hours during business hours.</p>
                <p>For urgent issues, please call:</p>
                <a href="tel:+16469782187" class="text-accent-500 hover:underline font-semibold">
                  +1 (646) 978-2187
                </a>
              </div>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
