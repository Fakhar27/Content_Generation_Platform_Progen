"use client"
import { useState, useEffect } from "react"
import { usePathname, useRouter } from "next/navigation"
import Link from "next/link"
import { useUser, useClerk } from "@clerk/nextjs"
import { Home, Video, ImageIcon, FileText, Settings, LogOut, Menu, X, LayoutDashboard, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { motion, AnimatePresence } from "framer-motion"

export function AppSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { user } = useUser()
  const { signOut } = useClerk()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isMobile && isMobileMenuOpen && !(event.target as Element).closest(".sidebar-wrapper")) {
        setIsMobileMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isMobile, isMobileMenuOpen]);

  const menuItems = [
    {
      name: "Home",
      href: "/",
      icon: Home,
    },
    {
      name: "Dashboard",
      href: "/dashboard",
      icon: LayoutDashboard,
    },
    {
      name: "Playground",
      href: "/playground",
      icon: Video,
      highlight: true,
    },
    {
      name: "Exports",
      href: "/exports",
      icon: Download,
    }
  ]

  const handleNavigation = (href: string) => {
    if (isMobile) {
      setIsMobileMenuOpen(false);
    }
    router.push(href);
    router.refresh();
  };

  const getInitials = () => {
    if (!user) return "?"
    const firstName = user.firstName || ""
    const lastName = user.lastName || ""
    if (firstName && lastName) {
      return `${firstName[0]}${lastName[0]}`
    } else if (firstName) {
      return firstName[0]
    } else if (user.username) {
      return user.username[0].toUpperCase()
    } else {
      return "U"
    }
  }

  return (
    <>
      {/* Mobile menu button */}
      <div className="fixed top-4 left-4 z-50 md:hidden">
        <Button
          variant="outline"
          size="icon"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="bg-zinc-900/80 backdrop-blur-sm border-zinc-800 hover:bg-zinc-800"
        >
          {isMobileMenuOpen ? <X className="h-5 w-5 text-white" /> : <Menu className="h-5 w-5 text-white" />}
        </Button>
      </div>

      {/* Mobile sidebar */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 bg-black/80 z-40 md:hidden"
              onClick={() => setIsMobileMenuOpen(false)}
            />
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 left-0 z-50 w-64 bg-black border-r border-zinc-800 md:hidden"
            >
              <div className="absolute inset-0 z-0 opacity-30">
                <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-pink-600/20 via-purple-600/10 to-transparent"></div>
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,0,255,0.15),transparent_50%)]"></div>
              </div>
              <div className="flex flex-col h-full relative z-10">
                <SidebarContent
                  menuItems={menuItems}
                  pathname={pathname}
                  onNavigate={handleNavigation}
                  user={user}
                  onSignOut={signOut}
                  getInitials={getInitials}
                />
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Desktop sidebar */}
      <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0 z-30">
        <div className="flex flex-col flex-1 min-h-0 bg-black border-r border-zinc-800 relative">
          <div className="absolute inset-0 z-0 opacity-30 pointer-events-none">
            <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-pink-600/20 via-purple-600/10 to-transparent"></div>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,0,255,0.15),transparent_50%)]"></div>
          </div>
          <div className="relative z-10">
            <SidebarContent
              menuItems={menuItems}
              pathname={pathname}
              onNavigate={handleNavigation}
              user={user}
              onSignOut={signOut}
              getInitials={getInitials}
            />
          </div>
        </div>
      </div>
    </>
  )
}

interface SidebarContentProps {
  menuItems: {
    name: string
    href: string
    icon: any
    highlight?: boolean
  }[]
  pathname: string
  onNavigate: (href: string) => void
  user: any
  onSignOut: () => void
  getInitials: () => string
}

function SidebarContent({ menuItems, pathname, onNavigate, user, onSignOut, getInitials }: SidebarContentProps) {
  return (
    <>
      <div className="flex items-center h-16 flex-shrink-0 px-4 border-b border-zinc-800">
        <Link
          href="/dashboard"
          className="flex items-center hover:opacity-80 transition-opacity w-full text-left"
        >
          <div className="w-8 h-8 rounded-md bg-gradient-to-br from-pink-600 to-purple-600 flex items-center justify-center mr-2">
            <Video className="h-4 w-4 text-white" />
          </div>
          <span className="text-xl font-bold text-white">Progen.AI</span>
        </Link>
      </div>
      <div className="flex flex-col h-[calc(100vh-4rem)]">
        <div className="flex-1 overflow-y-auto">
          <nav className="mt-5 flex-1 px-4 space-y-1">
            {menuItems.map((item) => {
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`
                    w-full group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all
                    ${
                      isActive
                        ? "bg-zinc-800 text-white border-l-2 border-pink-500"
                        : "text-zinc-400 hover:bg-zinc-800/50 hover:text-white"
                    }
                    ${item.highlight ? "relative overflow-hidden" : ""}
                    focus:outline-none focus:ring-2 focus:ring-pink-500/50
                    active:bg-zinc-800/70
                    select-none
                  `}
                  role="menuitem"
                  aria-current={isActive ? "page" : undefined}
                >
                  <div className="flex items-center w-full">
                    <item.icon
                      className={`mr-3 h-5 w-5 flex-shrink-0 ${
                        isActive ? "text-pink-500" : "text-zinc-500 group-hover:text-zinc-300"
                      }`}
                    />
                    <span className="flex-1">{item.name}</span>
                    {item.highlight && (
                      <span className="ml-auto px-2 py-0.5 text-xs rounded-full bg-pink-500/20 text-pink-500">New</span>
                    )}
                  </div>
                </Link>
              )
            })}
          </nav>
        </div>
        <div className="flex-shrink-0 border-t border-zinc-800 p-4">
          <div className="flex items-center w-full">
            <div className="flex-shrink-0">
              <Avatar>
                <AvatarImage src={user?.imageUrl} alt={user?.username || "User"} />
                <AvatarFallback className="bg-zinc-800 text-zinc-400">{getInitials()}</AvatarFallback>
              </Avatar>
            </div>
            <div className="ml-3 min-w-0 flex-1">
              <div className="text-sm font-medium text-white truncate">
                {user?.firstName ? `${user.firstName} ${user.lastName || ""}` : user?.username || "User"}
              </div>
              <div className="text-xs text-zinc-500 truncate">{user?.primaryEmailAddress?.emailAddress || ""}</div>
            </div>
            <Button 
              variant="ghost" 
              size="icon" 
              className="text-zinc-400 hover:text-white hover:bg-zinc-800" 
              onClick={onSignOut}
            >
              <LogOut className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}