"use client";

import { motion } from "framer-motion";
import { Clock } from "lucide-react";

interface StepDurationProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepDuration({ settings, onSettingsChange }: StepDurationProps) {
  const durationOptions = [
    { label: "20 seconds", value: 2, description: "Short and concise" },
    { label: "40 seconds", value: 3, description: "Standard length" },
    { label: "1 minute", value: 4, description: "Extended content" },
  ];

  return (
    <div className="max-w-xl mx-auto">
      <div className="flex flex-col space-y-4">
        <p className="text-zinc-400 text-sm mb-2">
          Select how long you want your video to be. Longer videos will include more content and detail.
        </p>
        
        <div className="grid grid-cols-1 gap-4">
          {durationOptions.map((option) => (
            <motion.div
              key={option.value}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`flex items-center p-4 rounded-lg cursor-pointer transition-colors ${
                settings.iterations === option.value
                  ? "bg-gradient-to-r from-purple-700 to-pink-700 border border-pink-500"
                  : "bg-zinc-900 border border-zinc-700 hover:border-zinc-500"
              }`}
              onClick={() => onSettingsChange("iterations", option.value)}
            >
              <div className={`p-2 rounded-full ${
                settings.iterations === option.value ? "bg-pink-500/30" : "bg-zinc-800"
              }`}>
                <Clock className={`h-5 w-5 ${
                  settings.iterations === option.value ? "text-pink-300" : "text-zinc-400"
                }`} />
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-white">{option.label}</h3>
                <p className={`text-sm ${
                  settings.iterations === option.value ? "text-pink-200" : "text-zinc-400"
                }`}>
                  {option.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}