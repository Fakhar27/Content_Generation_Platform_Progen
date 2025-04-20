"use client"

import { motion } from "framer-motion"
import { Loader2 } from "lucide-react"

export function LoadingScreen() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center"
    >
      <div className="text-center">
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
          className="inline-block mb-6"
        >
          <div className="relative">
            <Loader2 className="h-16 w-16 text-pink-500 animate-spin" />
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
          className="text-2xl font-bold mb-2"
        >
          GENERATING YOUR VIDEO
        </motion.h2>
        
        <p className="text-xs text-zinc-500 max-w-md italic">
          Our AI is crafting your video using the prompt and settings you provided.
          The complete video with synchronized subtitles will be available soon.
        </p>

        <div className="mt-8 h-2 w-128 bg-zinc-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ duration: 45 }} // Extended to account for longer generation time
            className="h-full bg-gradient-to-r from-purple-600 via-pink-600 to-red-600"
          ></motion.div>
        </div>
      </div>
    </motion.div>
  )
}