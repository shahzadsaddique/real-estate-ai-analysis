import Link from "next/link";
import { FileText, ArrowLeft } from "lucide-react";
import DocumentList from "@/components/DocumentList";

// For demo purposes, using a default user ID
// In production, this would come from authentication
const DEFAULT_USER_ID = "demo-user";

export default function DocumentsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      {/* Navigation */}
      <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <FileText className="h-6 w-6 text-primary" />
              <span className="text-xl font-bold">Real Estate AI</span>
            </div>
            <div className="flex items-center space-x-4">
              <Link
                href="/"
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors inline-flex items-center"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Home
              </Link>
              <Link
                href="/upload"
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                Upload
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-6xl mx-auto">
          <div className="space-y-8">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">
                Documents
              </h1>
              <p className="text-muted-foreground">
                View and manage your uploaded documents. Track processing status
                and access completed analyses.
              </p>
            </div>

            <div className="bg-card border rounded-lg p-6 shadow-sm">
              <DocumentList userId={DEFAULT_USER_ID} autoRefresh={true} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
