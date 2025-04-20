import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ClerkProvider } from "@clerk/nextjs";
import SidebarWrapper from "./(main)/_components/SidebarWrapper";// Import the new client component

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Progen.AI - AI Video Generator",
  description: "Generate punk-style AI videos with customization options",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <ClerkProvider>
      <html lang="en" className="dark">
        <body className={inter.className}>
        <div>
            <main>{children}</main>
          </div>
        </body>
      </html>
    </ClerkProvider>
  );
}
