/**
 * KYC ONBOARDING PAGE
 * Multi-step verification wizard for regulatory compliance
 * 
 * Features:
 * - 6-step progressive disclosure
 * - Real-time validation
 * - Document upload with preview
 * - Identity verification integration
 * - Trading experience assessment
 * - Legal agreements
 */

import { createSignal, createEffect, Show } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import {
  CheckCircle2,
  AlertCircle,
  User,
  Home,
  Briefcase,
  TrendingUp,
  FileText,
  Shield,
  ChevronRight,
  ChevronLeft,
  Upload,
} from 'lucide-solid';
import { apiClient, KYCProfile } from '../../lib/api/client';
import { PersonalInfoStep } from './steps/PersonalInfoStep';
import { AddressStep } from './steps/AddressStep';
import { EmploymentStep } from './steps/EmploymentStep';
import { TradingExperienceStep } from './steps/TradingExperienceStep';
import { DocumentsStep } from './steps/DocumentsStep';
import { AgreementsStep } from './steps/AgreementsStep';

type Step = 'personal' | 'address' | 'employment' | 'experience' | 'documents' | 'agreements';

const STEPS: { id: Step; label: string; icon: any }[] = [
  { id: 'personal', label: 'Personal Information', icon: User },
  { id: 'address', label: 'Address', icon: Home },
  { id: 'employment', label: 'Employment & Finances', icon: Briefcase },
  { id: 'experience', label: 'Trading Experience', icon: TrendingUp },
  { id: 'documents', label: 'Identity Documents', icon: FileText },
  { id: 'agreements', label: 'Agreements', icon: Shield },
];

export default function OnboardingPage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = createSignal<Step>('personal');
  const [profile, setProfile] = createSignal<Partial<KYCProfile>>({});
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [completedSteps, setCompletedSteps] = createSignal<Set<Step>>(new Set());

  createEffect(() => {
    loadProfile();
  });

  const loadProfile = async () => {
    try {
      const data = await apiClient.getKYCProfile();
      setProfile(data);
      
      // If already approved, redirect
      if (data.status === 'approved') {
        navigate('/dashboard');
      }
    } catch (err: any) {
      // Profile doesn't exist yet, that's ok
      if (err.status !== 404) {
        setError(err.message);
      }
    }
  };

  const currentStepIndex = () => STEPS.findIndex((s) => s.id === currentStep());

  const canProgress = () => {
    return completedSteps().has(currentStep());
  };

  const handleNext = async () => {
    if (!canProgress()) {
      setError('Please complete all required fields');
      return;
    }

    const nextIndex = currentStepIndex() + 1;
    if (nextIndex < STEPS.length) {
      setCurrentStep(STEPS[nextIndex].id);
      setError(null);
    } else {
      await handleSubmit();
    }
  };

  const handleBack = () => {
    const prevIndex = currentStepIndex() - 1;
    if (prevIndex >= 0) {
      setCurrentStep(STEPS[prevIndex].id);
      setError(null);
    }
  };

  const handleStepComplete = async (stepId: Step, data: Partial<KYCProfile>) => {
    // Optimistic update
    setProfile({ ...profile(), ...data });
    setCompletedSteps(new Set(completedSteps()).add(stepId));
    setError(null);
    
    // Save to backend
    try {
      await apiClient.updateKYCProfile(data);
    } catch (err: any) {
      console.error('Failed to save step:', err);
      // Don't block progress, but maybe show a warning?
      // For now, we assume it works or the final submit will catch missing data
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      await apiClient.submitKYCForReview();
      navigate('/onboarding/submitted');
    } catch (err: any) {
      setError(err.message || 'Failed to submit application');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="min-h-screen bg-terminal-950 flex items-center justify-center p-3 sm:p-4 md:p-6">
      <div class="max-w-4xl w-full">
        {/* Header */}
        <div class="text-center mb-4 sm:mb-6 md:mb-8">
          <h1 class="text-xl sm:text-2xl md:text-3xl font-bold text-white mb-2">Account Verification</h1>
          <p class="text-sm sm:text-base text-gray-400">
            Complete your profile to start trading. This process takes about 5-10 minutes.
          </p>
        </div>

        {/* Mobile Step Title */}
        <div class="sm:hidden text-center mb-4">
           <span class="text-accent-500 font-bold text-xs uppercase tracking-wider">Step {currentStepIndex() + 1} of {STEPS.length}</span>
           <h2 class="text-xl font-bold text-white mt-1">{STEPS[currentStepIndex()].label}</h2>
        </div>

        {/* Progress Bar */}
        <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-3 sm:p-4 md:p-6 mb-3 sm:mb-4 md:mb-6">
          <div class="flex items-center justify-between mb-4">
            {STEPS.map((step, index) => (
              <>
                <button
                  onClick={() => {
                    if (index <= currentStepIndex() || completedSteps().has(step.id)) {
                      setCurrentStep(step.id);
                    }
                  }}
                  class={`flex-1 flex flex-col items-center gap-2 transition-colors ${
                    index <= currentStepIndex() ? 'cursor-pointer' : 'cursor-not-allowed'
                  }`}
                  disabled={index > currentStepIndex() && !completedSteps().has(step.id)}
                >
                  <div
                    class={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                      completedSteps().has(step.id)
                        ? 'bg-success-500 border-success-500'
                        : currentStep() === step.id
                        ? 'bg-accent-500 border-accent-500'
                        : index < currentStepIndex()
                        ? 'bg-terminal-850 border-primary-500'
                        : 'bg-terminal-850 border-terminal-750'
                    }`}
                  >
                    {completedSteps().has(step.id) ? (
                      <CheckCircle2 size={16} class="text-white sm:w-5 sm:h-5" />
                    ) : (
                      <step.icon
                        size={16}
                        class="sm:w-5 sm:h-5"
                        classList={{
                          'text-white': currentStep() === step.id,
                          'text-primary-500': index <= currentStepIndex() && currentStep() !== step.id,
                          'text-gray-600': index > currentStepIndex() && currentStep() !== step.id
                        }}
                      />
                    )}
                  </div>
                  <div
                    class={`text-[10px] sm:text-xs font-semibold text-center hidden sm:block ${
                      currentStep() === step.id
                        ? 'text-white'
                        : completedSteps().has(step.id)
                        ? 'text-success-500'
                        : index <= currentStepIndex()
                        ? 'text-gray-400'
                        : 'text-gray-600'
                    }`}
                  >
                    {step.label}
                  </div>
                </button>
                {index < STEPS.length - 1 && (
                  <div
                    class={`h-0.5 flex-1 mx-2 transition-colors ${
                      index < currentStepIndex() || completedSteps().has(STEPS[index + 1].id)
                        ? 'bg-primary-500'
                        : 'bg-terminal-750'
                    }`}
                  />
                )}
              </>
            ))}
          </div>

          <div class="flex items-center justify-between text-[10px] sm:text-xs text-gray-500">
            <span>
              Step {currentStepIndex() + 1} of {STEPS.length}
            </span>
            <span class="hidden sm:inline">{Math.round(((currentStepIndex() + 1) / STEPS.length) * 100)}% Complete</span>
            <span class="sm:hidden">{Math.round(((currentStepIndex() + 1) / STEPS.length) * 100)}%</span>
          </div>
        </div>

        {/* Error Banner */}
        <Show when={error()}>
          <div class="bg-danger-500/10 border border-danger-500/20 rounded-lg p-3 sm:p-4 mb-3 sm:mb-4 md:mb-6 flex items-center justify-between gap-2 sm:gap-3">
            <div class="flex items-center gap-2 sm:gap-3">
              <AlertCircle size={16} class="sm:w-5 sm:h-5 text-danger-500 flex-shrink-0" />
              <span class="text-xs sm:text-sm text-danger-500">{error()}</span>
            </div>
            <button 
              onClick={() => loadProfile()}
              class="px-3 py-1.5 bg-danger-900/50 hover:bg-danger-900 text-danger-200 text-xs font-medium rounded border border-danger-800 transition-colors whitespace-nowrap"
            >
              Retry
            </button>
          </div>
        </Show>

        {/* Step Content */}
        <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-4 sm:p-6 md:p-8 mb-3 sm:mb-4 md:mb-6">
          <Show when={currentStep() === 'personal'}>
            <PersonalInfoStep
              profile={profile()}
              onComplete={(data) => handleStepComplete('personal', data)}
            />
          </Show>
          <Show when={currentStep() === 'address'}>
            <AddressStep
              profile={profile()}
              onComplete={(data) => handleStepComplete('address', data)}
            />
          </Show>
          <Show when={currentStep() === 'employment'}>
            <EmploymentStep
              profile={profile()}
              onComplete={(data) => handleStepComplete('employment', data)}
            />
          </Show>
          <Show when={currentStep() === 'experience'}>
            <TradingExperienceStep
              profile={profile()}
              onComplete={(data) => handleStepComplete('experience', data)}
            />
          </Show>
          <Show when={currentStep() === 'documents'}>
            <DocumentsStep
              profile={profile()}
              onComplete={(data) => handleStepComplete('documents', data)}
            />
          </Show>
          <Show when={currentStep() === 'agreements'}>
            <AgreementsStep
              profile={profile()}
              onComplete={(data) => handleStepComplete('agreements', data)}
            />
          </Show>
        </div>

        {/* Navigation */}
        <div class="flex items-center justify-between gap-2 sm:gap-4">
          <button
            onClick={handleBack}
            disabled={currentStepIndex() === 0}
            class="flex items-center gap-1 sm:gap-2 px-3 sm:px-4 md:px-6 py-2 sm:py-2.5 md:py-3 bg-terminal-900 hover:bg-terminal-850 disabled:opacity-50 disabled:cursor-not-allowed border border-terminal-750 text-white text-sm sm:text-base font-semibold rounded-lg transition-colors"
          >
            <ChevronLeft size={16} class="sm:w-5 sm:h-5" />
            <span>Back</span>
          </button>

          <button
            onClick={handleNext}
            disabled={!canProgress() || loading()}
            class="flex items-center gap-1 sm:gap-2 px-3 sm:px-4 md:px-6 py-2 sm:py-2.5 md:py-3 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm sm:text-base font-semibold rounded-lg transition-colors"
          >
            <span>
              {currentStepIndex() === STEPS.length - 1
                ? loading()
                  ? 'Submitting...'
                  : <><span class="hidden sm:inline">Submit Application</span><span class="sm:hidden">Submit</span></>
                : 'Continue'}
            </span>
            {currentStepIndex() < STEPS.length - 1 && <ChevronRight size={16} class="sm:w-5 sm:h-5" />}
          </button>
        </div>

        {/* Security Notice */}
        <div class="mt-4 sm:mt-6 md:mt-8 p-3 sm:p-4 bg-primary-500/5 border border-primary-500/20 rounded-lg">
          <div class="flex items-start gap-2 sm:gap-3">
            <Shield size={16} class="sm:w-5 sm:h-5 text-primary-500 flex-shrink-0 mt-0.5" />
            <div class="text-[10px] sm:text-xs text-gray-400">
              <span class="font-semibold text-primary-500 block mb-1">Your Information is Secure</span>
              All data is encrypted and transmitted securely. We comply with all SEC and FINRA regulations
              for customer identification and verification. Your information will never be sold or shared
              with third parties for marketing purposes.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
