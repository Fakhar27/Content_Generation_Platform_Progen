"use client";

import { useState } from "react";
import { Sparkles } from "lucide-react";

interface StepPromptProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepPrompt({ prompt, onPromptChange, onPromptSubmit }: StepPromptProps) {
  const [isFocused, setIsFocused] = useState(false);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (prompt.trim().length > 0) {
        onPromptSubmit(prompt);
      }
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="relative">
        <div
          className={`relative border ${
            isFocused ? "border-pink-500" : "border-zinc-700"
          } bg-zinc-900 rounded-xl p-3 transition-all`}
        >
          <div className="absolute top-3 left-3 px-2 py-1 bg-pink-500 text-xs text-white font-bold rounded-full flex items-center">
            <Sparkles className="h-3 w-3 mr-1" />
            AI Powered
          </div>
          <div className="pt-10 pb-2 px-2">
            <h3 className="text-lg font-semibold text-white mb-1">Video Prompt</h3>
            <p className="text-sm text-zinc-400 mb-3">Describe your video idea in detail. Be creative and specific!</p>
            <textarea
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Enter your video idea here..."
              className="w-full h-40 bg-zinc-950 border border-zinc-800 rounded-lg p-4 text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-pink-500 transition-all resize-none"
            />
            <div className="flex justify-end mt-2">
              <span className="text-xs text-zinc-500">{prompt.length} / 500</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}