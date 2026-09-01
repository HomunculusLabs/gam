import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "GAM Codebase City";
const description =
  "A living procedural 3D city of the GAM codebase: files, crates, issues, comments, tests, commits, CI, failures, success, interconnections, local measurement lag, and evolution through time.";
const origin = new URL("https://gam-codebase-city.sauerslabs.chatgpt.site");

export const metadata: Metadata = {
  metadataBase: origin,
  title,
  description,
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  openGraph: {
    title,
    description,
    type: "website",
    url: origin,
    siteName: title,
    images: [
      {
        url: "/og.png",
        width: 1672,
        height: 941,
        alt: "A bright procedural city generated from the GAM repository",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
