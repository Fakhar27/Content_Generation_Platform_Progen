import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: {
    buildActivity: false,
    buildActivityPosition: "bottom-right", // Change position if needed
  }, // ✅ Fixed closing bracket and semicolon her
};

export default nextConfig;
