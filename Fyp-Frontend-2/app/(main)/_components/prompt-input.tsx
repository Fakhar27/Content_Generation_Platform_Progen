"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Sparkles, ArrowRight } from "lucide-react"

interface PromptInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
}

export function PromptInput({ value, onChange, onSubmit }: PromptInputProps) {
  const [isFocused, setIsFocused] = useState(false)

  const handleSubmit = () => {
    if (value.trim()) {
      onSubmit(value)
    }
  }

  return (
    <motion.div
      className={`relative rounded-xl border ${
        isFocused ? "border-pink-500 shadow-[0_0_15px_rgba(255,0,255,0.3)]" : "border-zinc-800"
      } bg-zinc-900 p-6 transition-all duration-300`}
      animate={{ scale: isFocused ? 1.02 : 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white mb-2">Video Prompt</h2>
        <p className="text-zinc-400 text-sm">Describe your video idea in detail. Be creative and specific!</p>
      </div>

      <div className="relative">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder="A child playing in playground looking up in the sky and seeing a shooting star"
          className="min-h-[150px] bg-zinc-950 border-zinc-800 text-white placeholder:text-zinc-600 focus:border-pink-500 focus:ring-pink-500 resize-none"
        />

        <div className="absolute bottom-3 right-3">
          <span className="text-xs text-zinc-500 mr-2">{value.length} / 500</span>
        </div>
      </div>

      <div className="mt-6 flex justify-end">
        <Button
          onClick={handleSubmit}
          disabled={!value.trim()}
          className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold"
        >
          Generate Video <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>

      <div className="absolute -top-2 -left-2">
        <div className="bg-pink-600 text-white text-xs px-2 py-1 rounded-md flex items-center">
          <Sparkles className="h-3 w-3 mr-1" />
          AI Powered
        </div>
      </div>
    </motion.div>
  )
}
