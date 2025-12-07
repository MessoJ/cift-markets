import { createSignal, createEffect, For, Show } from 'solid-js';
import { Upload, File, Trash2, CheckCircle2, AlertCircle } from 'lucide-solid';
import { apiClient, KYCProfile, KYCDocument } from '../../../lib/api/client';

interface DocumentsStepProps {
  profile: Partial<KYCProfile>;
  onComplete: (data: Partial<KYCProfile>) => void;
}

export function DocumentsStep(props: DocumentsStepProps) {
  const [documents, setDocuments] = createSignal<KYCDocument[]>([]);
  const [uploading, setUploading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  createEffect(() => {
    loadDocuments();
  });

  const loadDocuments = async () => {
    try {
      const docs = await apiClient.getKYCDocuments();
      setDocuments(docs);
      
      // Mark complete if required docs uploaded
      const hasID = docs.some((d) => 
        d.document_type === 'drivers_license' || d.document_type === 'passport' || d.document_type === 'national_id'
      );
      const hasProof = docs.some((d) => d.document_type === 'proof_of_address');
      
      if (hasID && hasProof) {
        props.onComplete({ documents_uploaded: true });
      }
    } catch (err: any) {
      if (err.status !== 404) {
        setError(err.message);
      }
    }
  };

  const handleFileUpload = async (file: File, docType: string) => {
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setUploading(true);
    setError(null);
    try {
      await apiClient.uploadKYCDocument(file, docType);
      await loadDocuments();
    } catch (err: any) {
      setError(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await apiClient.deleteKYCDocument(docId);
      await loadDocuments();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const DocumentUploader = (props: {
    type: string;
    label: string;
    description: string;
    required?: boolean;
  }) => {
    const uploadedDoc = () => documents().find((d) => d.document_type === props.type);

    return (
      <div class="p-3 sm:p-4 bg-terminal-850 border border-terminal-750 rounded">
        <div class="flex items-start justify-between mb-2 sm:mb-3">
          <div>
            <div class="text-xs sm:text-sm font-semibold text-white mb-1">
              {props.label}
              {props.required && <span class="text-danger-500 ml-1">*</span>}
            </div>
            <div class="text-[10px] sm:text-xs text-gray-400">{props.description}</div>
          </div>
          <Show when={uploadedDoc()}>
            <CheckCircle2 size={16} class="sm:w-5 sm:h-5 text-success-500" />
          </Show>
        </div>

        <Show when={!uploadedDoc()}>
          <label class="block">
            <input
              type="file"
              accept="image/*,.pdf"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file, props.type);
              }}
              class="hidden"
            />
            <div class="flex items-center justify-center gap-2 px-3 sm:px-4 py-2 sm:py-3 bg-terminal-900 hover:bg-terminal-800 border border-terminal-750 hover:border-accent-500 text-accent-500 rounded cursor-pointer transition-colors">
              <Upload size={14} class="sm:w-4 sm:h-4" />
              <span class="text-xs sm:text-sm font-semibold">Choose File</span>
            </div>
          </label>
        </Show>

        <Show when={uploadedDoc()}>
          <div class="flex items-center gap-3 p-3 bg-terminal-900 border border-terminal-750 rounded">
            <File size={16} class="text-gray-400" />
            <div class="flex-1 min-w-0">
              <div class="text-sm text-white truncate">{uploadedDoc()!.file_name}</div>
              <div class="text-xs text-gray-500">
                {(uploadedDoc()!.file_size / 1024).toFixed(1)} KB • 
                Uploaded {new Date(uploadedDoc()!.uploaded_at).toLocaleDateString()}
              </div>
            </div>
            <button
              onClick={() => handleDelete(uploadedDoc()!.id)}
              class="p-2 hover:bg-terminal-800 text-danger-500 rounded transition-colors"
            >
              <Trash2 size={16} />
            </button>
          </div>
        </Show>
      </div>
    );
  };

  return (
    <div>
      <h2 class="text-lg sm:text-xl font-bold text-white mb-2">Identity Documents</h2>
      <p class="text-xs sm:text-sm text-gray-400 mb-4 sm:mb-6">Upload documents to verify your identity</p>

      <Show when={error()}>
        <div class="mb-3 sm:mb-4 p-2 sm:p-3 bg-danger-500/10 border border-danger-500/20 rounded flex items-center gap-2">
          <AlertCircle size={16} class="text-danger-500" />
          <span class="text-xs sm:text-sm text-danger-500">{error()}</span>
        </div>
      </Show>

      <div class="space-y-3 sm:space-y-4">
        <div class="p-3 sm:p-4 bg-warning-500/5 border border-warning-500/20 rounded">
          <div class="text-xs sm:text-sm font-semibold text-warning-500 mb-2">Document Requirements</div>
          <ul class="text-[10px] sm:text-xs text-gray-400 space-y-1 list-disc list-inside">
            <li>Documents must be clear and legible</li>
            <li>All four corners must be visible</li>
            <li>Documents must be valid (not expired)</li>
            <li>Accepted formats: JPG, PNG, PDF (max 10MB)</li>
          </ul>
        </div>

        <DocumentUploader
          type="drivers_license"
          label="Government-Issued ID"
          description="Driver's license, passport, or national ID card"
          required
        />

        <DocumentUploader
          type="proof_of_address"
          label="Proof of Address"
          description="Utility bill, bank statement, or lease agreement (dated within 90 days)"
          required
        />

        <DocumentUploader
          type="tax_document"
          label="Tax Document (Optional)"
          description="W-9 form or other tax identification document"
        />

        <Show when={uploading()}>
          <div class="p-3 sm:p-4 bg-primary-500/5 border border-primary-500/20 rounded text-center">
            <div class="text-xs sm:text-sm text-primary-500 font-semibold">Uploading document...</div>
          </div>
        </Show>

        <div class="p-3 sm:p-4 bg-primary-500/5 border border-primary-500/20 rounded text-[10px] sm:text-xs text-gray-400">
          <span class="font-semibold text-primary-500">Privacy & Security:</span> Your documents are
          encrypted using bank-level 256-bit SSL encryption. We use automated identity verification
          technology to verify your documents. Documents are securely stored and only accessible to
          authorized compliance personnel.
        </div>
      </div>
    </div>
  );
}
