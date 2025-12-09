import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Real Estate AI Analysis Platform",
  description:
    "Production-ready platform for analyzing real estate documents using AI. Upload zoning maps, risk assessments, and permits to generate comprehensive property analysis.",
  keywords: [
    "real estate",
    "AI analysis",
    "property analysis",
    "zoning",
    "risk assessment",
    "document processing",
  ],
  authors: [{ name: "Real Estate AI Platform" }],
  openGraph: {
    title: "Real Estate AI Analysis Platform",
    description:
      "Analyze real estate documents with AI-powered property analysis",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
