"use client";

import { motion } from "framer-motion";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface StepGenreProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepGenre({ settings, onSettingsChange }: StepGenreProps) {
  const genres = [
    { value: "Adventure", icon: "🌄", description: "Exciting journeys and explorations" },
    { value: "Horror", icon: "👻", description: "Scary and suspenseful content" },
    { value: "Fantasy", icon: "🧙‍♂️", description: "Magical and mythical elements" },
    { value: "Funny", icon: "😂", description: "Humorous and comedic content" },
    { value: "Children", icon: "🧸", description: "Kid-friendly and educational" },
    { value: "Documentary", icon: "🎬", description: "Informative and factual" },
    { value: "Cinematic", icon: "🎞️", description: "Movie-like with dramatic elements" },
    { value: "Vlog", icon: "📹", description: "Personal and casual style" },
  ];

  return (
    <div className="max-w-xl mx-auto">
      <div className="grid grid-cols-2 gap-4">
        {genres.map((genre) => (
          <motion.div
            key={genre.value}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            className={`flex flex-col p-4 rounded-lg cursor-pointer transition-colors border ${
              settings.genre === genre.value
                ? "border-pink-500 bg-pink-500/10"
                : "border-zinc-700 bg-zinc-900 hover:border-zinc-500"
            }`}
            onClick={() => onSettingsChange("genre", genre.value)}
          >
            <div className="flex items-center mb-2">
              <span className="text-2xl mr-2">{genre.icon}</span>
              <span className="text-lg font-medium text-white">{genre.value}</span>
            </div>
            <p className="text-sm text-zinc-400">{genre.description}</p>
          </motion.div>
        ))}
      </div>

      {/* Mobile-friendly alternative */}
      <div className="mt-6 md:hidden">
        <label className="block text-sm font-medium text-white mb-2">Select Genre</label>
        <Select value={settings.genre} onValueChange={(value) => onSettingsChange("genre", value)}>
          <SelectTrigger className="bg-zinc-950 border-zinc-800 text-white">
            <SelectValue placeholder="Select a genre" />
          </SelectTrigger>
          <SelectContent className="bg-zinc-950 border-zinc-800 text-white">
            {genres.map((genre) => (
              <SelectItem key={genre.value} value={genre.value} className="capitalize">
                <div className="flex items-center">
                  <span className="mr-2">{genre.icon}</span>
                  {genre.value}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}