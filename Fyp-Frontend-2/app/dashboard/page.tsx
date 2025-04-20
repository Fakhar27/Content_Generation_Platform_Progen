import { DashboardHeader } from "../(main)/_components/dashboard/dashboard-header";
import { DashboardShell } from "../(main)/_components/dashboard/dashboard-shell";
import { StatsCards } from "../(main)/_components/dashboard/stats-cards";
import { RecentProjects } from "../(main)/_components/dashboard/recent-projects";
import { CreateNewButton } from "../(main)/_components/dashboard/create-new-button";
import SidebarWrapper from "../(main)/_components/SidebarWrapper";

export default function DashboardPage() {
  return (
    <div className="bg-zinc-900 flex min-h-screen w-full">
      <div className="w-64">
        <SidebarWrapper />
      </div>
      <div className="flex-1 w-full">
        <DashboardShell>
          <div className="flex flex-col min-h-screen w-full">
            <DashboardHeader
              heading="Dashboard"
              subheading="Manage your AI video projects"
            >
              <CreateNewButton />
            </DashboardHeader>
            <div className="mt-10">
              <RecentProjects />
            </div>
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

