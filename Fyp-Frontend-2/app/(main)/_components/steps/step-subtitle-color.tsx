"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Type, Check } from "lucide-react";

interface StepSubtitleColorProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepSubtitleColor({ settings, onSettingsChange }: StepSubtitleColorProps) {
  const [showCustom, setShowCustom] = useState(false);
  
  const presetColors = [
    { label: "Neon Green", value: "#00ff66" },
    { label: "Hot Pink", value: "#ff00ff" },
    { label: "Electric Blue", value: "#00ffff" },
    { label: "Bright Yellow", value: "#ffff00" },
    { label: "Pure White", value: "#ffffff" },
    { label: "Vibrant Orange", value: "#ff6600" },
  ];

  const handleColorChange = (color: string) => {
    onSettingsChange("subtitleColor", color);
  };

  return (
    <div className="max-w-xl mx-auto">
      <div className="flex flex-col space-y-6">
        <p className="text-zinc-400 text-sm mb-2">
          Choose a color for your video subtitles. This will determine how your text appears in the video.
        </p>
        
        {/* Color preview */}
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800">
          <h3 className="text-lg font-medium mb-3 text-white">Preview</h3>
          <div className="h-20 rounded-lg bg-zinc-800 flex items-center justify-center">
            <div className="px-4 py-2 rounded bg-zinc-900/80 backdrop-blur-sm">
              <span style={{ color: settings.subtitleColor }} className="text-xl font-medium">
                Sample subtitle text
              </span>
            </div>
          </div>
        </div>
        
        {/* Color preset buttons */}
        <div>
          <h3 className="text-lg font-medium mb-3 text-white">Preset Colors</h3>
          <div className="grid grid-cols-3 gap-3">
            {presetColors.map((color) => (
              <motion.div
                key={color.value}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`flex flex-col items-center p-3 rounded-lg cursor-pointer border ${
                  settings.subtitleColor === color.value 
                    ? "border-pink-500" 
                    : "border-zinc-700 hover:border-zinc-500"
                }`}
                onClick={() => handleColorChange(color.value)}
              >
                <div 
                  className="w-10 h-10 rounded-full mb-2 flex items-center justify-center" 
                  style={{ backgroundColor: color.value }}
                >
                  {settings.subtitleColor === color.value && (
                    <Check className="h-5 w-5 text-black" />
                  )}
                </div>
                <span className="text-xs text-zinc-300">{color.label}</span>
              </motion.div>
            ))}
          </div>
        </div>
        
        {/* Custom color picker */}
        <div className="p-4 rounded-lg border border-zinc-700 bg-zinc-900/50">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-white">Custom Color</h3>
            <button
              onClick={() => setShowCustom(!showCustom)}
              className="text-sm text-pink-400 hover:text-pink-300"
            >
              {showCustom ? "Hide" : "Show"}
            </button>
          </div>
          
          {showCustom && (
            <div className="mt-3">
              <div className="flex items-center gap-4">
                <Input
                  type="color"
                  value={settings.subtitleColor}
                  onChange={(e) => handleColorChange(e.target.value)}
                  className="w-16 h-10 p-1 bg-transparent border border-zinc-700 rounded-md"
                />
                <Input
                  type="text"
                  value={settings.subtitleColor}
                  onChange={(e) => handleColorChange(e.target.value)}
                  className="bg-zinc-950 border-zinc-800 text-white"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}