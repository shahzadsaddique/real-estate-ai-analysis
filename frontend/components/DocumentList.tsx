"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  FileText,
  Download,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  AlertCircle,
  RefreshCw,
  FileSearch,
} from "lucide-react";
import {
  getDocument,
  getDocumentStatus,
  listDocuments,
  deleteDocument,
  DocumentListItem,
  DocumentStatusResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import AnalysisViewer from "./AnalysisViewer";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";

interface DocumentListProps {
  userId: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onDocumentSelect?: (documentId: string) => void;
  onDocumentDelete?: (documentId: string) => void;
  className?: string;
}

type DocumentWithStatus = DocumentListItem & {
  statusData?: DocumentStatusResponse;
  signedUrl?: string;
};

const STATUS_CONFIG: Record<
  string,
  { label: string; color: string; icon: React.ReactNode }
> = {
  uploaded: {
    label: "Uploaded",
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
    icon: <Clock className="h-3 w-3" />,
  },
  processing: {
    label: "Processing",
    color:
      "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
  },
  parsing: {
    label: "Parsing",
    color:
      "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
  },
  chunking: {
    label: "Chunking",
    color:
      "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
  },
  indexing: {
    label: "Indexing",
    color:
      "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
  },
  indexed: {
    label: "Indexed",
    color:
      "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
    icon: <CheckCircle2 className="h-3 w-3" />,
  },
  analyzing: {
    label: "Analyzing",
    color:
      "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
  },
  complete: {
    label: "Complete",
    color:
      "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    icon: <CheckCircle2 className="h-3 w-3" />,
  },
  failed: {
    label: "Failed",
    color:
      "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
    icon: <XCircle className="h-3 w-3" />,
  },
};

const DOCUMENT_TYPE_COLORS: Record<string, string> = {
  zoning: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  risk: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  permit: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  other: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300",
};

export default function DocumentList({
  userId,
  autoRefresh = true,
  refreshInterval = 10000, // 10 seconds
  onDocumentSelect,
  onDocumentDelete,
  className,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentWithStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [downloadingIds, setDownloadingIds] = useState<Set<string>>(new Set());
  const [viewingAnalysisId, setViewingAnalysisId] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const statusUpdateRefs = useRef<Map<string, NodeJS.Timeout>>(new Map());

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatDate = (dateString: string): string => {
    // Parse the date string - if it doesn't have timezone info, treat it as UTC
    let dateStr = dateString.trim();
    
    // Check if it already has timezone info (Z, +, or - after the time)
    const hasTimezone = /[+-]\d{2}:\d{2}$/.test(dateStr) || dateStr.endsWith('Z');
    
    // If no timezone info, assume it's UTC and append 'Z'
    if (!hasTimezone && dateStr.includes('T')) {
      dateStr = dateStr + 'Z';
    }
    
    const date = new Date(dateStr);
    
    // Validate the date
    if (isNaN(date.getTime())) {
      console.warn(`Invalid date string: ${dateString}`);
      return "Unknown";
    }
    
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
    });
  };

  const updateDocumentStatus = useCallback(
    async (documentId: string) => {
      try {
        const statusData = await getDocumentStatus(documentId);
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.id === documentId
              ? {
                  ...doc,
                  status: statusData.status,
                  statusData,
                  error_message: statusData.error_message,
                  progress: statusData.progress,
                }
              : doc
          )
        );
        return statusData;
      } catch (err: any) {
        console.error(`Failed to update status for ${documentId}:`, err);
        return null;
      }
    },
    []
  );

  const fetchDocuments = useCallback(async () => {
    try {
      setError(null);
      const docs = await listDocuments(userId);
      const docsWithStatus: DocumentWithStatus[] = docs.map((doc) => ({
        ...doc,
      }));

      setDocuments(docsWithStatus);

      // Fetch status for all documents in parallel
      const statusPromises = docsWithStatus.map((doc) =>
        updateDocumentStatus(doc.id)
      );
      await Promise.all(statusPromises);
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        "Failed to load documents";
      setError(errorMessage);
      console.error("Failed to fetch documents:", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [userId, updateDocumentStatus]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchDocuments();
  }, [fetchDocuments]);


  const handleDownload = useCallback(
    async (documentId: string) => {
      if (downloadingIds.has(documentId)) return;

      setDownloadingIds((prev) => new Set(prev).add(documentId));
      try {
        // Get document metadata to get filename
        const doc = await getDocument(documentId, userId, false);
        
        // Download file from API
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "https://realestate-api-784415003538.us-central1.run.app/api/v1"}/documents/${documentId}/download?user_id=${userId}`,
          {
            method: "GET",
          }
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: "Failed to download document" }));
          throw new Error(errorData.detail || "Failed to download document");
        }

        // Get blob from response
        const blob = await response.blob();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = doc.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.detail ||
          err.message ||
          "Failed to download document";
        console.error("Error downloading document:", err);
        alert(`Failed to download document: ${errorMessage}`);
      } finally {
        setDownloadingIds((prev) => {
          const next = new Set(prev);
          next.delete(documentId);
          return next;
        });
      }
    },
    [userId, downloadingIds]
  );

  const handleDelete = useCallback(
    async (documentId: string) => {
      if (!confirm("Are you sure you want to delete this document?")) {
        return;
      }

      setDeletingIds((prev) => new Set(prev).add(documentId));
      try {
        await deleteDocument(documentId, userId);
        setDocuments((prev) => prev.filter((doc) => doc.id !== documentId));
        if (onDocumentDelete) {
          onDocumentDelete(documentId);
        }
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.detail ||
          err.message ||
          "Failed to delete document";
        alert(errorMessage);
      } finally {
        setDeletingIds((prev) => {
          const next = new Set(prev);
          next.delete(documentId);
          return next;
        });
      }
    },
    [userId, onDocumentDelete]
  );

  // Set up polling for documents that are still processing
  useEffect(() => {
    if (!autoRefresh) return;

    const processingDocs = documents.filter(
      (doc) =>
        doc.status !== "complete" &&
        doc.status !== "failed" &&
        doc.status !== "error"
    );

    // Clear existing intervals
    statusUpdateRefs.current.forEach((interval) => clearInterval(interval));
    statusUpdateRefs.current.clear();

    // Set up polling for each processing document
    processingDocs.forEach((doc) => {
      const interval = setInterval(() => {
        updateDocumentStatus(doc.id);
      }, refreshInterval);
      statusUpdateRefs.current.set(doc.id, interval);
    });

    return () => {
      statusUpdateRefs.current.forEach((interval) => clearInterval(interval));
      statusUpdateRefs.current.clear();
    };
  }, [documents, autoRefresh, refreshInterval, updateDocumentStatus]);

  // Initial fetch
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      statusUpdateRefs.current.forEach((interval) => clearInterval(interval));
      statusUpdateRefs.current.clear();
    };
  }, []);

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center p-8", className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">
          Loading documents...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn("p-4", className)}>
        <div className="flex items-start space-x-2 p-4 bg-destructive/10 border border-destructive/20 rounded-md">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-destructive">Error</p>
            <p className="text-sm text-destructive/80">{error}</p>
            <button
              onClick={handleRefresh}
              className="mt-2 text-sm text-destructive underline hover:no-underline"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className={cn("p-8 text-center", className)}>
        <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-sm font-medium text-muted-foreground">
          No documents found
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          Upload your first document to get started
        </p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Documents</h2>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground h-9 px-3"
        >
          <RefreshCw
            className={cn(
              "h-4 w-4 mr-2",
              refreshing && "animate-spin"
            )}
          />
          Refresh
        </button>
      </div>

      {/* Document List */}
      <div className="space-y-2">
        {documents.map((doc) => {
          const statusConfig =
            STATUS_CONFIG[doc.status] ||
            STATUS_CONFIG["uploaded"];
          // Check both top-level and metadata for document_type
          const docType = doc.document_type || doc.metadata?.document_type || "other";
          const typeColor = DOCUMENT_TYPE_COLORS[docType] || DOCUMENT_TYPE_COLORS.other;
          const isDeleting = deletingIds.has(doc.id);
          const isDownloading = downloadingIds.has(doc.id);
          const progressPercent =
            doc.statusData?.progress?.progress_percent ||
            doc.progress?.progress_percent ||
            0;

          return (
            <div
              key={doc.id}
              className="border rounded-lg p-4 hover:bg-accent/50 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-2">
                    <FileText className="h-5 w-5 text-primary flex-shrink-0" />
                    <h3 className="text-sm font-medium truncate">
                      {doc.filename}
                    </h3>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    {/* Status Badge */}
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                        statusConfig.color
                      )}
                    >
                      {statusConfig.icon}
                      {statusConfig.label}
                    </span>

                    {/* Document Type Badge */}
                    <span
                      className={cn(
                        "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium capitalize",
                        typeColor
                      )}
                    >
                      {docType}
                    </span>

                    {/* File Size */}
                    <span className="text-xs text-muted-foreground">
                      {formatFileSize(doc.file_size ?? doc.metadata?.file_size ?? 0)}
                    </span>
                  </div>

                  {/* Document Metadata (when available) */}
                  {(doc.metadata?.page_count != null ||
                    doc.metadata?.tables_count != null ||
                    doc.metadata?.images_count != null ||
                    doc.metadata?.text_blocks_count != null) && (
                    <div className="flex flex-wrap items-center gap-3 mb-2 text-xs text-muted-foreground">
                      {doc.metadata?.page_count != null && (
                        <span>
                          <span className="font-medium">{doc.metadata.page_count}</span>{" "}
                          {doc.metadata.page_count === 1 ? "page" : "pages"}
                        </span>
                      )}
                      {doc.metadata?.tables_count != null && (
                        <span>
                          <span className="font-medium">{doc.metadata.tables_count}</span>{" "}
                          {doc.metadata.tables_count === 1 ? "table" : "tables"}
                        </span>
                      )}
                      {doc.metadata?.images_count != null && (
                        <span>
                          <span className="font-medium">{doc.metadata.images_count}</span>{" "}
                          {doc.metadata.images_count === 1 ? "image" : "images"}
                        </span>
                      )}
                      {doc.metadata?.text_blocks_count != null && (
                        <span>
                          <span className="font-medium">{doc.metadata.text_blocks_count}</span>{" "}
                          {doc.metadata.text_blocks_count === 1 ? "text block" : "text blocks"}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Progress Bar for Processing Documents */}
                  {doc.status !== "complete" &&
                    doc.status !== "failed" &&
                    progressPercent > 0 && (
                      <div className="mb-2">
                        <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                          <span>{doc.statusData?.progress?.stage || doc.status}</span>
                          <span>{progressPercent}%</span>
                        </div>
                        <div className="w-full bg-secondary rounded-full h-1.5">
                          <div
                            className="bg-primary h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${progressPercent}%` }}
                          />
                        </div>
                      </div>
                    )}

                  {/* Error Message */}
                  {doc.error_message && (
                    <div className="mt-2 text-xs text-destructive">
                      {doc.error_message}
                    </div>
                  )}

                  {/* Timestamp */}
                  <div className="text-xs text-muted-foreground mt-2">
                    Uploaded {formatDate(doc.created_at)}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-1 ml-4 flex-shrink-0">
                  {(doc.status === "complete" || doc.status === "indexed") && (
                    <button
                      onClick={() => setViewingAnalysisId(doc.id)}
                      disabled={isDeleting}
                      className="inline-flex items-center justify-center rounded-md p-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground"
                      title="View analysis"
                    >
                      <FileSearch className="h-4 w-4" />
                    </button>
                  )}
                  <button
                    onClick={() => handleDownload(doc.id)}
                    disabled={isDeleting || isDownloading || doc.status !== "complete"}
                    className="inline-flex items-center justify-center rounded-md p-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground"
                    title="Download document"
                  >
                    {isDownloading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Download className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    disabled={isDeleting}
                    className="inline-flex items-center justify-center rounded-md p-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-destructive/10 hover:text-destructive"
                    title="Delete document"
                  >
                    {isDeleting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Analysis Viewer Dialog */}
      <Dialog open={viewingAnalysisId !== null} onOpenChange={(open) => !open && setViewingAnalysisId(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Document Analysis</DialogTitle>
          </DialogHeader>
          {viewingAnalysisId && (
            <AnalysisViewer
              documentId={viewingAnalysisId}
              documentType={
                documents.find((d) => d.id === viewingAnalysisId)?.document_type ||
                documents.find((d) => d.id === viewingAnalysisId)?.metadata?.document_type
              }
              autoRefresh={true}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
