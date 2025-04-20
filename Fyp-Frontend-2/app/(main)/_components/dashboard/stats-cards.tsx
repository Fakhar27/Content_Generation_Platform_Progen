"use client"

import { motion } from "framer-motion"
import { Video, Clock, Users, TrendingUp } from "lucide-react"

const stats = [
  {
    title: "Total Videos",
    value: "0",
    icon: Video,
    color: "pink",
    change: "+12%",
    trend: "up",
  },
  {
    title: "Processing",
    value: "3",
    icon: Clock,
    color: "cyan",
    change: "-2",
    trend: "down",
  },
  {
    title: "Viewers",
    value: "0",
    icon: Users,
    color: "purple",
    change: "+18%",
    trend: "up",
  },
  {
    title: "Engagement",
    value: "0%",
    icon: TrendingUp,
    color: "green",
    change: "+5%",
    trend: "up",
  },
]

export function StatsCards() {
  return (
    <>
      {stats.map((stat, index) => (
        <motion.div
          key={stat.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
          className="punk-card p-6"
        >
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm text-zinc-400">{stat.title}</p>
              <h3 className={`text-3xl font-bold mt-2 text-neon-${stat.color}`}>{stat.value}</h3>
            </div>
            <div className={`p-2 rounded-lg bg-${stat.color}-500/10`}>
              <stat.icon className={`h-5 w-5 text-${stat.color}-500`} />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <span className={`${stat.trend === "up" ? "text-green-500" : "text-red-500"} font-medium mr-1`}>
              {stat.change}
            </span>
            <span className="text-zinc-400">from last month</span>
          </div>
        </motion.div>
      ))}
    </>
  )
}