"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"

interface GlitchTextProps {
  children: React.ReactNode
  className?: string
}

export function GlitchText({ children, className = "" }: GlitchTextProps) {
  const [isGlitching, setIsGlitching] = useState(false)

  useEffect(() => {
    // Randomly trigger glitch effect
    const interval = setInterval(
      () => {
        setIsGlitching(true)
        setTimeout(() => setIsGlitching(false), 200)
      },
      Math.random() * 5000 + 3000,
    )

    return () => clearInterval(interval)
  }, [])

  return (
    <div className={`relative inline-block ${className}`}>
      <div className="relative z-10">{children}</div>

      {isGlitching && (
        <>
          <motion.div
            className="absolute inset-0 text-cyan-500 z-0"
            animate={{ x: [-2, 1, -1, 0], opacity: [0.8, 0.5, 0.8] }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
          <motion.div
            className="absolute inset-0 text-pink-500 z-0"
            animate={{ x: [2, -1, 1, 0], opacity: [0.8, 0.5, 0.8] }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
        </>
      )}
    </div>
  )
}