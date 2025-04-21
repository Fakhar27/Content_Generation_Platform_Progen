"use client"

import { motion } from "framer-motion"
import { Loader2 } from "lucide-react"

export function LoadingScreen() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-4 sm:p-6 md:p-8"
    >
      <div className="text-center max-w-md mx-auto">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 15, -15, 0],
          }}
          transition={{
            duration: 2,
            repeat: Number.POSITIVE_INFINITY,
            repeatType: "loop",
          }}
          className="inline-block mb-4 sm:mb-6"
        >
          <div className="relative">
            <Loader2 className="h-12 w-12 sm:h-16 sm:w-16 text-pink-500 animate-spin" />
            <div className="absolute inset-0 blur-lg opacity-70 bg-pink-500 rounded-full"></div>
          </div>
        </motion.div>

        <motion.h2
          animate={{
            color: ["#ff00ff", "#00ffff", "#ff00ff"],
          }}
          transition={{
            duration: 3,
            repeat: Number.POSITIVE_INFINITY,
          }}
          className="text-xl sm:text-2xl font-bold mb-2 sm:mb-4"
        >
          GENERATING YOUR VIDEO
        </motion.h2>
        
        <p className="text-xs sm:text-sm text-zinc-500 max-w-md italic mb-4 sm:mb-6">
          Our AI is crafting your video using the prompt and settings you provided.
          The complete video with synchronized subtitles will be available soon.
        </p>

        <div className="h-1.5 sm:h-2 w-48 sm:w-64 md:w-80 bg-zinc-800 rounded-full overflow-hidden mx-auto">
          <motion.div
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ duration: 45 }}
            className="h-full bg-gradient-to-r from-purple-600 via-pink-600 to-red-600"
          ></motion.div>
        </div>
      </div>
    </motion.div>
  )
}