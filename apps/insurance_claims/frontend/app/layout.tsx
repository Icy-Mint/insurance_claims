import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Insurance Claims Support",
  description: "SOP-guided conversational agent for insurance claims support",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
