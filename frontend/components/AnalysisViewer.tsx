"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Download,
  Printer,
  FileText,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
} from "lucide-react";
import { getAnalysis, generateAnalysis, AnalysisResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface AnalysisViewerProps {
  documentId: string;
  documentType?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onAnalysisComplete?: (analysis: AnalysisResponse) => void;
  className?: string;
}

interface ExpandableSectionProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  icon?: React.ReactNode;
}

function ExpandableSection({
  title,
  children,
  defaultExpanded = false,
  icon,
}: ExpandableSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-accent/50 transition-colors text-left"
      >
        <div className="flex items-center space-x-2">
          {icon && <span className="text-primary">{icon}</span>}
          <h3 className="font-semibold">{title}</h3>
        </div>
        {isExpanded ? (
          <ChevronDown className="h-5 w-5 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-5 w-5 text-muted-foreground" />
        )}
      </button>
      {isExpanded && (
        <div className="p-4 border-t bg-muted/30">{children}</div>
      )}
    </div>
  );
}

function JSONViewer({ data }: { data: any }) {
  const [copied, setCopied] = useState(false);
  const jsonString = JSON.stringify(data, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative">
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={handleCopy}
          className="px-2 py-1 text-xs bg-background border rounded hover:bg-accent transition-colors"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="p-4 bg-slate-950 dark:bg-slate-900 text-slate-100 rounded-md overflow-x-auto text-sm font-mono">
        <code>{jsonString}</code>
      </pre>
    </div>
  );
}

function ListSection({ items, title }: { items: string[]; title: string }) {
  if (!items || items.length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="font-medium text-sm text-muted-foreground">{title}</h4>
      <ul className="space-y-1">
        {items.map((item, index) => (
          <li key={index} className="flex items-start space-x-2 text-sm">
            <span className="text-primary mt-1.5">•</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function KeyValueSection({
  data,
  title,
}: {
  data: Record<string, any>;
  title: string;
}) {
  if (!data || Object.keys(data).length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="font-medium text-sm text-muted-foreground">{title}</h4>
      <div className="space-y-2">
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="flex flex-col sm:flex-row sm:items-center">
            <span className="font-medium text-sm sm:w-1/3 capitalize">
              {key.replace(/_/g, " ")}:
            </span>
            <span className="text-sm sm:w-2/3">
              {typeof value === "object" ? (
                <JSONViewer data={value} />
              ) : (
                String(value)
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AnalysisViewer({
  documentId,
  documentType,
  autoRefresh = true,
  refreshInterval = 5000,
  onAnalysisComplete,
  className,
}: AnalysisViewerProps) {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchAnalysis = useCallback(async () => {
    try {
      setError(null);
      const data = await getAnalysis(documentId);

      // Handle 202 Accepted (processing)
      if (data.status === "processing" || data.status === "pending") {
        setAnalysis(data);
        setLoading(false);
        setRefreshing(false);
        return;
      }

      setAnalysis(data);
      setLoading(false);
      setRefreshing(false);

      if (data.status === "complete" && data.result && onAnalysisComplete) {
        onAnalysisComplete(data);
      }
    } catch (err: any) {
      if (err.response?.status === 404) {
        // Analysis doesn't exist yet
        setAnalysis(null);
        setError(null);
      } else {
        const errorMessage =
          err.response?.data?.detail ||
          err.message ||
          "Failed to load analysis";
        setError(errorMessage);
      }
      setLoading(false);
      setRefreshing(false);
    }
  }, [documentId, onAnalysisComplete]);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setError(null);
    try {
      await generateAnalysis(documentId, documentType);
      // Wait a moment then fetch
      await new Promise((resolve) => setTimeout(resolve, 1000));
      await fetchAnalysis();
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        "Failed to generate analysis";
      setError(errorMessage);
    } finally {
      setGenerating(false);
    }
  }, [documentId, documentType, fetchAnalysis]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchAnalysis();
  }, [fetchAnalysis]);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  const handleExport = useCallback(() => {
    if (!analysis || !analysis.result) return;

    const dataStr = JSON.stringify(analysis.result, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `analysis_${documentId}_${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [analysis, documentId]);

  // Set up polling for processing analyses
  useEffect(() => {
    if (!autoRefresh) return;

    const shouldPoll =
      analysis &&
      (analysis.status === "processing" || analysis.status === "pending");

    if (shouldPoll) {
      intervalRef.current = setInterval(() => {
        fetchAnalysis();
      }, refreshInterval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [analysis, autoRefresh, refreshInterval, fetchAnalysis]);

  // Initial fetch
  useEffect(() => {
    fetchAnalysis();
  }, [fetchAnalysis]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center p-8", className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">
          Loading analysis...
        </span>
      </div>
    );
  }

  if (error && !analysis) {
    return (
      <div className={cn("p-4", className)}>
        <div className="flex items-start space-x-2 p-4 bg-destructive/10 border border-destructive/20 rounded-md">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-destructive">Error</p>
            <p className="text-sm text-destructive/80">{error}</p>
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="mt-2 inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
            >
              {generating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Analysis"
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className={cn("p-8 text-center", className)}>
        <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-sm font-medium text-muted-foreground mb-2">
          No analysis found
        </p>
        <p className="text-xs text-muted-foreground mb-4">
          Generate an analysis to view results
        </p>
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
        >
          {generating ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <FileText className="mr-2 h-4 w-4" />
              Generate Analysis
            </>
          )}
        </button>
      </div>
    );
  }

  const isProcessing = analysis.status === "processing" || analysis.status === "pending";
  const isComplete = analysis.status === "complete";
  const isFailed = analysis.status === "failed";

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {isProcessing && (
            <>
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <span className="text-sm font-medium">Processing Analysis...</span>
            </>
          )}
          {isComplete && (
            <>
              <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium">Analysis Complete</span>
            </>
          )}
          {isFailed && (
            <>
              <XCircle className="h-5 w-5 text-destructive" />
              <span className="text-sm font-medium">Analysis Failed</span>
            </>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground h-9 px-3"
          >
            <RefreshCw
              className={cn("h-4 w-4", refreshing && "animate-spin")}
            />
          </button>
          {isComplete && analysis.result && (
            <>
              <button
                onClick={handlePrint}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 hover:bg-accent hover:text-accent-foreground h-9 px-3"
              >
                <Printer className="h-4 w-4 mr-2" />
                Print
              </button>
              <button
                onClick={handleExport}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 hover:bg-accent hover:text-accent-foreground h-9 px-3"
              >
                <Download className="h-4 w-4 mr-2" />
                Export JSON
              </button>
            </>
          )}
        </div>
      </div>

      {/* Error Message */}
      {isFailed && analysis.error_message && (
        <div className="flex items-start space-x-2 p-4 bg-destructive/10 border border-destructive/20 rounded-md">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-destructive">Analysis Failed</p>
            <p className="text-sm text-destructive/80">{analysis.error_message}</p>
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="mt-2 text-sm text-destructive underline hover:no-underline"
            >
              {generating ? "Generating..." : "Retry Analysis"}
            </button>
          </div>
        </div>
      )}

      {/* Processing Status */}
      {isProcessing && (
        <div className="flex items-center space-x-2 p-4 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-md">
          <Clock className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <div>
            <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
              Analysis in Progress
            </p>
            <p className="text-xs text-blue-700 dark:text-blue-300">
              This may take a few minutes. The page will update automatically.
            </p>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {isComplete && analysis.result && (
        <div className="space-y-4">
          {/* Property Information */}
          {analysis.result.analysis && (
            <>
              {/* Basic Property Info */}
              {(analysis.result.analysis.property_address ||
                analysis.result.analysis.property_type ||
                analysis.result.analysis.zoning_classification) && (
                <ExpandableSection
                  title="Property Information"
                  defaultExpanded={true}
                  icon={<FileText className="h-4 w-4" />}
                >
                  <div className="space-y-3">
                    {analysis.result.analysis.property_address && (
                      <div>
                        <span className="text-sm font-medium">Address: </span>
                        <span className="text-sm">
                          {analysis.result.analysis.property_address}
                        </span>
                      </div>
                    )}
                    {analysis.result.analysis.property_type && (
                      <div>
                        <span className="text-sm font-medium">Type: </span>
                        <span className="text-sm">
                          {analysis.result.analysis.property_type}
                        </span>
                      </div>
                    )}
                    {analysis.result.analysis.zoning_classification && (
                      <div>
                        <span className="text-sm font-medium">
                          Zoning Classification:{" "}
                        </span>
                        <span className="text-sm">
                          {analysis.result.analysis.zoning_classification}
                        </span>
                      </div>
                    )}
                    {analysis.result.analysis.compliance_status && (
                      <div>
                        <span className="text-sm font-medium">
                          Compliance Status:{" "}
                        </span>
                        <span
                          className={cn(
                            "text-sm font-medium",
                            analysis.result.analysis.compliance_status.toLowerCase().includes("compliant")
                              ? "text-green-600 dark:text-green-400"
                              : "text-yellow-600 dark:text-yellow-400"
                          )}
                        >
                          {analysis.result.analysis.compliance_status}
                        </span>
                      </div>
                    )}
                  </div>
                </ExpandableSection>
              )}

              {/* Zoning Summary */}
              {analysis.result.analysis.zoning_summary && (
                <ExpandableSection
                  title="Zoning Summary"
                  defaultExpanded={true}
                  icon={<FileText className="h-4 w-4" />}
                >
                  <p className="text-sm whitespace-pre-wrap">
                    {analysis.result.analysis.zoning_summary}
                  </p>
                </ExpandableSection>
              )}

              {/* Key Findings */}
              {analysis.result.analysis.key_findings &&
                analysis.result.analysis.key_findings.length > 0 && (
                  <ExpandableSection
                    title="Key Findings"
                    defaultExpanded={true}
                    icon={<CheckCircle2 className="h-4 w-4" />}
                  >
                    <ListSection
                      items={analysis.result.analysis.key_findings}
                      title=""
                    />
                  </ExpandableSection>
                )}

              {/* Permit Requirements */}
              {analysis.result.analysis.permit_requirements &&
                analysis.result.analysis.permit_requirements.length > 0 && (
                  <ExpandableSection
                    title="Permit Requirements"
                    icon={<FileText className="h-4 w-4" />}
                  >
                    <ListSection
                      items={analysis.result.analysis.permit_requirements}
                      title=""
                    />
                  </ExpandableSection>
                )}

              {/* Restrictions */}
              {analysis.result.analysis.restrictions &&
                analysis.result.analysis.restrictions.length > 0 && (
                  <ExpandableSection
                    title="Restrictions"
                    icon={<AlertCircle className="h-4 w-4" />}
                  >
                    <ListSection
                      items={analysis.result.analysis.restrictions}
                      title=""
                    />
                  </ExpandableSection>
                )}

              {/* Recommendations */}
              {analysis.result.analysis.recommendations &&
                analysis.result.analysis.recommendations.length > 0 && (
                  <ExpandableSection
                    title="Recommendations"
                    defaultExpanded={true}
                    icon={<CheckCircle2 className="h-4 w-4" />}
                  >
                    <ListSection
                      items={analysis.result.analysis.recommendations}
                      title=""
                    />
                  </ExpandableSection>
                )}

              {/* Risk Assessment */}
              {analysis.result.analysis.risk_assessment && (
                <ExpandableSection
                  title="Risk Assessment"
                  icon={<AlertCircle className="h-4 w-4" />}
                >
                  <KeyValueSection
                    data={analysis.result.analysis.risk_assessment}
                    title=""
                  />
                </ExpandableSection>
              )}

              {/* Additional Insights */}
              {analysis.result.analysis.additional_insights && (
                <ExpandableSection
                  title="Additional Insights"
                  icon={<FileText className="h-4 w-4" />}
                >
                  <KeyValueSection
                    data={analysis.result.analysis.additional_insights}
                    title=""
                  />
                </ExpandableSection>
              )}
            </>
          )}

          {/* Analysis Metadata */}
          <ExpandableSection
            title="Analysis Metadata"
            icon={<FileText className="h-4 w-4" />}
          >
            <div className="space-y-2 text-sm">
              {analysis.result.confidence_score !== undefined && 
               analysis.result.confidence_score !== null && (
                <div>
                  <span className="font-medium">Confidence Score: </span>
                  <span>
                    {(analysis.result.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {analysis.result.processing_time_seconds !== undefined && 
               analysis.result.processing_time_seconds !== null && (
                <div>
                  <span className="font-medium">Processing Time: </span>
                  <span>
                    {analysis.result.processing_time_seconds.toFixed(2)} seconds
                  </span>
                </div>
              )}
              {analysis.result.llm_model && (
                <div>
                  <span className="font-medium">Model Used: </span>
                  <span>{analysis.result.llm_model}</span>
                </div>
              )}
              {analysis.result.tokens_used !== undefined && 
               analysis.result.tokens_used !== null && (
                <div>
                  <span className="font-medium">Tokens Used: </span>
                  <span>{analysis.result.tokens_used.toLocaleString()}</span>
                </div>
              )}
              {analysis.result.chunks_retrieved !== undefined && 
               analysis.result.chunks_retrieved !== null && (
                <div>
                  <span className="font-medium">Chunks Retrieved: </span>
                  <span>{analysis.result.chunks_retrieved}</span>
                </div>
              )}
              {analysis.result.source_documents &&
                analysis.result.source_documents.length > 0 && (
                  <div>
                    <span className="font-medium">Source Documents: </span>
                    <span>{analysis.result.source_documents.join(", ")}</span>
                  </div>
                )}
              {analysis.completed_at && (
                <div>
                  <span className="font-medium">Completed At: </span>
                  <span>
                    {new Date(analysis.completed_at).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </ExpandableSection>

          {/* Raw JSON View */}
          <ExpandableSection
            title="Raw JSON Data"
            icon={<FileText className="h-4 w-4" />}
          >
            <JSONViewer data={analysis.result} />
          </ExpandableSection>
        </div>
      )}
    </div>
  );
}
