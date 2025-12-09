import Link from "next/link";
import { FileText, Upload, BarChart3, Zap } from "lucide-react";

export default function HomePage() {
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
                href="/documents"
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                Documents
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

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Real Estate AI Analysis Platform
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Transform complex property documents into actionable insights using
              advanced AI and machine learning.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex items-center justify-center gap-4 pt-4">
            <Link
              href="/upload"
              className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <Upload className="mr-2 h-4 w-4" />
              Upload Document
            </Link>
            <Link
              href="/documents"
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              View Documents
            </Link>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 pt-16">
            <div className="space-y-3 text-center">
              <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 mx-auto">
                <FileText className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Intelligent Parsing</h3>
              <p className="text-sm text-muted-foreground">
                Advanced PDF parsing with layout awareness. Preserves tables,
                images, and multi-column layouts for accurate analysis.
              </p>
            </div>

            <div className="space-y-3 text-center">
              <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 mx-auto">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">AI-Powered Analysis</h3>
              <p className="text-sm text-muted-foreground">
                Generate comprehensive property analysis using Vertex AI and
                LangChain. Synthesize information from multiple documents.
              </p>
            </div>

            <div className="space-y-3 text-center">
              <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 mx-auto">
                <Zap className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Fast Processing</h3>
              <p className="text-sm text-muted-foreground">
                Async document processing with real-time status updates. Handle
                large documents efficiently with intelligent chunking.
              </p>
            </div>
          </div>

          {/* How It Works */}
          <div className="pt-16 space-y-8">
            <h2 className="text-2xl font-bold text-center">How It Works</h2>
            <div className="grid md:grid-cols-4 gap-6">
              <div className="space-y-2 text-center">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground font-bold mx-auto">
                  1
                </div>
                <h4 className="font-semibold">Upload</h4>
                <p className="text-sm text-muted-foreground">
                  Upload PDF documents (zoning maps, risk assessments, permits)
                </p>
              </div>
              <div className="space-y-2 text-center">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground font-bold mx-auto">
                  2
                </div>
                <h4 className="font-semibold">Process</h4>
                <p className="text-sm text-muted-foreground">
                  AI parses and chunks documents while preserving structure
                </p>
              </div>
              <div className="space-y-2 text-center">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground font-bold mx-auto">
                  3
                </div>
                <h4 className="font-semibold">Analyze</h4>
                <p className="text-sm text-muted-foreground">
                  Generate comprehensive property analysis using LLM orchestration
                </p>
              </div>
              <div className="space-y-2 text-center">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground font-bold mx-auto">
                  4
                </div>
                <h4 className="font-semibold">Review</h4>
                <p className="text-sm text-muted-foreground">
                  View structured analysis results with key findings and recommendations
                </p>
              </div>
            </div>
          </div>

          {/* Tech Stack */}
          <div className="pt-16 space-y-4">
            <h2 className="text-2xl font-bold text-center">Built With</h2>
            <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
              <span className="px-3 py-1 rounded-md bg-muted">Next.js 14</span>
              <span className="px-3 py-1 rounded-md bg-muted">FastAPI</span>
              <span className="px-3 py-1 rounded-md bg-muted">Vertex AI</span>
              <span className="px-3 py-1 rounded-md bg-muted">LangChain</span>
              <span className="px-3 py-1 rounded-md bg-muted">Pinecone</span>
              <span className="px-3 py-1 rounded-md bg-muted">Google Cloud</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-sm text-muted-foreground">
            <p>Real Estate AI Analysis Platform - Demonstrating expertise in AI, GCP, and LLM orchestration</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
