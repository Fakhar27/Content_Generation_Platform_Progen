"use client"; // Ensure this is a client component

import { usePathname } from "next/navigation";
import { AppSidebar } from "./AppSidebar";
import Image from "next/image";

const hiddenRoutes = ["/sign-in", "/sign-up"];

export default function SidebarWrapper() {
  const pathname = usePathname();

  if (hiddenRoutes.includes(pathname)) {
    return null; // Don't render sidebar on hidden routes
  }

  return <AppSidebar />;
}