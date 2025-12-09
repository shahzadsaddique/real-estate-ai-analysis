"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, FileText, X, CheckCircle2, AlertCircle } from "lucide-react";
import { uploadDocument, DocumentUploadResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface DocumentUploadProps {
  userId: string;
  onUploadSuccess?: (response: DocumentUploadResponse) => void;
  onUploadError?: (error: string) => void;
  className?: string;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_TYPES = ["application/pdf"];

export default function DocumentUpload({
  userId,
  onUploadSuccess,
  onUploadError,
  className,
}: DocumentUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<DocumentUploadResponse | null>(null);
  const [documentType, setDocumentType] = useState<string>("other");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Only PDF files are allowed";
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      return `File size exceeds ${MAX_FILE_SIZE / (1024 * 1024)}MB limit`;
    }

    if (file.size === 0) {
      return "File is empty";
    }

    return null;
  };

  const handleFileSelect = useCallback(
    (file: File) => {
      setError(null);
      setSuccess(null);

      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        setSelectedFile(null);
        return;
      }

      setSelectedFile(file);
    },
    []
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFileSelect(file);
      }
    },
    [handleFileSelect]
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFileSelect(file);
      }
    },
    [handleFileSelect]
  );

  const handleUpload = useCallback(async () => {
    if (!selectedFile) {
      setError("Please select a file first");
      return;
    }

    setIsUploading(true);
    setError(null);
    setSuccess(null);
    setUploadProgress(0);

    try {
      const response = await uploadDocument(
        selectedFile,
        userId,
        documentType, // Always send document type, including "other"
        (progress) => {
          setUploadProgress(progress);
        }
      );

      setSuccess(response);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        "Failed to upload document. Please try again.";
      setError(errorMessage);

      if (onUploadError) {
        onUploadError(errorMessage);
      }
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  }, [selectedFile, userId, documentType, onUploadSuccess, onUploadError]);

  const handleRemoveFile = useCallback(() => {
    setSelectedFile(null);
    setError(null);
    setSuccess(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Document Type Selection */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Document Type</label>
        <select
          value={documentType}
          onChange={(e) => setDocumentType(e.target.value)}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          disabled={isUploading}
        >
          <option value="other">Other</option>
          <option value="zoning">Zoning</option>
          <option value="risk">Risk Assessment</option>
          <option value="permit">Permit</option>
        </select>
      </div>

      {/* Drag and Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-lg p-8 text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50",
          isUploading && "opacity-50 cursor-not-allowed"
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileInputChange}
          className="hidden"
          disabled={isUploading}
        />

        {!selectedFile ? (
          <div className="space-y-4">
            <div className="flex flex-col items-center justify-center space-y-2">
              <Upload className="h-12 w-12 text-muted-foreground" />
              <div className="space-y-1">
                <p className="text-sm font-medium">
                  Drag and drop a PDF file here, or click to browse
                </p>
                <p className="text-xs text-muted-foreground">
                  Maximum file size: {MAX_FILE_SIZE / (1024 * 1024)}MB
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
            >
              <Upload className="mr-2 h-4 w-4" />
              Select File
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-muted rounded-md">
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-primary" />
                <div className="text-left">
                  <p className="text-sm font-medium">{selectedFile.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
              {!isUploading && (
                <button
                  type="button"
                  onClick={handleRemoveFile}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>

            {/* Upload Progress */}
            {isUploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Uploading...</span>
                  <span className="font-medium">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Upload Button */}
            {!isUploading && (
              <button
                type="button"
                onClick={handleUpload}
                className="w-full inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <Upload className="mr-2 h-4 w-4" />
                Upload Document
              </button>
            )}
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-start space-x-2 p-4 bg-destructive/10 border border-destructive/20 rounded-md">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-destructive">Upload Failed</p>
            <p className="text-sm text-destructive/80">{error}</p>
          </div>
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div className="flex items-start space-x-2 p-4 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800 rounded-md">
          <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-green-800 dark:text-green-200">
              Upload Successful
            </p>
            <p className="text-sm text-green-700 dark:text-green-300">
              {success.message}
            </p>
            <p className="text-xs text-green-600 dark:text-green-400 mt-1">
              Document ID: {success.document_id}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
