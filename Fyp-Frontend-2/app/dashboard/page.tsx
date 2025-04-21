"use client"

import { DashboardHeader } from "../(main)/_components/dashboard/dashboard-header";
import { DashboardShell } from "../(main)/_components/dashboard/dashboard-shell";
import { RecentProjects } from "../(main)/_components/dashboard/recent-projects";
import { CreateNewButton } from "../(main)/_components/dashboard/create-new-button";
import { AppSidebar } from "../(main)/_components/AppSidebar";

export default function DashboardPage() {
  return (
    <div className="flex min-h-screen w-full bg-black">
      <AppSidebar />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col w-full md:pl-64">
        <div className="relative">
          <div className="absolute inset-0 z-0 opacity-30">
            <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-pink-600/20 via-purple-600/10 to-transparent"></div>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,0,255,0.15),transparent_50%)]"></div>
          </div>
          <DashboardShell>
            <div className="flex flex-col min-h-screen w-full relative z-10">
              <DashboardHeader
                heading="Dashboard"
                subheading="Manage your AI video projects"
              >
                <div className="mt-4 sm:mt-0">
                  <CreateNewButton />
                </div>
              </DashboardHeader>
              
              {/* Recent Projects Container with responsive spacing */}
              <div className="mt-6 sm:mt-8 lg:mt-10 px-4 sm:px-6 pb-8">
                <RecentProjects />
              </div>
            </div>
          </DashboardShell>
        </div>
      </div>
    </div>
  );
}

