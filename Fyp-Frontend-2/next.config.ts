// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Configure output as standalone for Docker deployment
  output: "standalone",
  
  // Disable development indicators
  devIndicators: {
    buildActivity: false,
    buildActivityPosition: "bottom-right",
  },
  
  // Configure for AWS ECS deployment
  // This ensures the app can be run behind a load balancer
  poweredByHeader: false,
  
  // Configure image domains if needed
  images: {
    domains: [
      "addjrawhfmcnodhlqnkk.supabase.co",
      // Add additional domains if needed for your images
    ],
  },
  
  // Configure async storage for NextAuth
  experimental: {
    serverComponentsExternalPackages: ["@prisma/client"],
  },
  
  // Redirect trailing slashes
  async redirects() {
    return [
      {
        source: "/:path+/",
        destination: "/:path+",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;