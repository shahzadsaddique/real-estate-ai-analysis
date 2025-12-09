import axios, { AxiosError, AxiosProgressEvent } from "axios";

// API base URL - uses environment variable or defaults to localhost
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "https://realestate-api-784415003538.us-central1.run.app/api/v1";

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// ============================================================================
// Types
// ============================================================================

export interface DocumentListItem {
  id: string;
  user_id: string;
  filename: string;
  status: string;
  document_type?: string;
  file_size?: number;
  created_at: string;
  updated_at?: string;
  metadata?: {
    document_type?: string;
    file_size?: number;
    page_count?: number;
    tables_count?: number;
    images_count?: number;
    text_blocks_count?: number;
    [key: string]: any;
  };
  error_message?: string;
  progress?: {
    stage?: string;
    progress_percent?: number;
  };
}

export interface DocumentStatusResponse {
  document_id: string;
  status: string;
  progress?: {
    stage: string;
    progress_percent: number;
  };
  error_message?: string;
  updated_at: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  status: string;
  message: string;
  storage_path: string;
}

export interface DocumentResponse extends DocumentListItem {
  signed_url?: string;
  chunks?: any[];
}

export interface AnalysisResponse {
  analysis_id: string;
  document_id: string;
  status: "pending" | "processing" | "complete" | "failed";
  result?: {
    analysis?: {
      property_address?: string;
      property_type?: string;
      zoning_classification?: string;
      compliance_status?: string;
      zoning_summary?: string;
      key_findings?: string[];
      permit_requirements?: string[];
      restrictions?: string[];
      recommendations?: string[];
      risk_assessment?: Record<string, any>;
      additional_insights?: Record<string, any>;
      [key: string]: any;
    };
    confidence_score?: number;
    processing_time_seconds?: number;
    llm_model?: string;
    tokens_used?: number;
    chunks_retrieved?: number;
    source_documents?: string[];
    [key: string]: any;
  };
  created_at: string;
  updated_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface GenerateAnalysisResponse {
  analysis_id: string;
  document_id: string;
  status: string;
  message: string;
}

// ============================================================================
// Document API Functions
// ============================================================================

/**
 * Upload a document
 */
export async function uploadDocument(
  file: File,
  userId: string,
  documentType: string = "other",
  onProgress?: (progress: number) => void
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("user_id", userId);
  formData.append("document_type", documentType);

  try {
    const response = await apiClient.post<DocumentUploadResponse>(
      "/documents/upload",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (progressEvent.total && onProgress) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(percentCompleted);
          }
        },
      }
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw error;
    }
    throw new Error("Failed to upload document");
  }
}

/**
 * Get a document by ID
 */
export async function getDocument(
  documentId: string,
  userId: string,
  includeChunks: boolean = false
): Promise<DocumentResponse> {
  try {
    const response = await apiClient.get<DocumentResponse>(
      `/documents/${documentId}`,
      {
        params: {
          user_id: userId,
          include_chunks: includeChunks,
        },
      }
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw error;
    }
    throw new Error("Failed to get document");
  }
}

/**
 * Get document status
 */
export async function getDocumentStatus(
  documentId: string
): Promise<DocumentStatusResponse> {
  try {
    const response = await apiClient.get<DocumentStatusResponse>(
      `/documents/${documentId}/status`
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw error;
    }
    throw new Error("Failed to get document status");
  }
}

/**
 * List documents for a user
 */
export async function listDocuments(
  userId: string,
  limit: number = 50
): Promise<DocumentListItem[]> {
  try {
    const response = await apiClient.get<DocumentListItem[]>("/documents", {
      params: {
        user_id: userId,
        limit,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw error;
    }
    throw new Error("Failed to list documents");
  }
}

/**
 * Delete a document
 */
export async function deleteDocument(
  documentId: string,
  userId: string
): Promise<void> {
  try {
    await apiClient.delete(`/documents/${documentId}`, {
      params: {
        user_id: userId,
      },
    });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw error;
    }
    throw new Error("Failed to delete document");
  }
}

// ============================================================================
// Analysis API Functions
// ============================================================================

/**
 * Get analysis for a document
 */
export async function getAnalysis(
  documentId: string
): Promise<AnalysisResponse> {
  try {
    const response = await apiClient.get<AnalysisResponse>(
      `/documents/${documentId}/analysis`
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.response?.status === 404) {
        // Analysis doesn't exist yet - return a response indicating this
        throw axiosError;
      }
      throw error;
    }
    throw new Error("Failed to get analysis");
  }
}

/**
 * Generate analysis for a document
 */
export async function generateAnalysis(
  documentId: string,
  documentType?: string
): Promise<GenerateAnalysisResponse> {
  try {
    const response = await apiClient.post<GenerateAnalysisResponse>(
      "/generate",
      {
        document_id: documentId,
        document_type: documentType || "other",
      }
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw error;
    }
    throw new Error("Failed to generate analysis");
  }
}
