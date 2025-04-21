"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Volume2, Play, Pause, VolumeX, Check } from "lucide-react";

interface StepMusicProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepMusic({ settings, onSettingsChange }: StepMusicProps) {
  const [playingMusic, setPlayingMusic] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  // Initialize audio element
  useEffect(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
    }
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);
  
  const musicOptions = [
    {
      value: "1",
      label: "Experiance",
      description: "Beautiful rendition of Ludovico Einaudi",
      sample: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/audio/Audio/Experiance.mp3", //ADD SUPASBASE URL HERE
    },
    {
      value: "2",
      label: "Soul",
      description: "Perfect with an adventrous reel",
      sample: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/audio/Audio/BM.wav", 
    },
    {
      value: "3",
      label: "Aadat",
      description: "Nescafe Basement's Aadat",
      sample: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/audio/Audio/Aadat.mp3", 
    },
    {
      value: "0",
      label: "No Music",
      description: "Generate video without background music",
      sample: "",
    },
  ];

  // Audio setup is now handled in the main useEffect

  const togglePlayback = (music: string, sample: string) => {
    if (!sample) return;
    
    try {
      if (playingMusic === music) {
        if (audioRef.current) {
          audioRef.current.pause();
        }
        setPlayingMusic(null);
      } else {
        if (!audioRef.current) {
          audioRef.current = new Audio(sample);
        } else {
          audioRef.current.src = sample;
        }
        
        const playPromise = audioRef.current.play();
        
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              setPlayingMusic(music);
            })
            .catch(err => {
              console.error("Failed to play audio:", err);
              // Try alternative playback method for some browsers
              setTimeout(() => {
                if (audioRef.current) {
                  audioRef.current.play()
                    .then(() => setPlayingMusic(music))
                    .catch(e => console.error("Second attempt failed:", e));
                }
              }, 100);
            });
        }
      }
    } catch (error) {
      console.error("Audio playback error:", error);
    }
  };

  return (
    <div className="max-w-xl mx-auto">
      <div className="flex flex-col space-y-4">
        <p className="text-zinc-400 text-sm mb-2">
          Choose background music for your video or select "No Music" if you prefer just voice narration.
        </p>
        
        <audio 
          ref={audioRef} 
          className="hidden" 
          onEnded={() => setPlayingMusic(null)}
        />
        
        <div className="grid grid-cols-1 gap-4">
          {musicOptions.map((option) => (
            <motion.div
              key={option.value}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`flex items-center p-4 rounded-lg cursor-pointer transition-colors ${
                settings.backgroundMusic === option.value
                  ? "bg-gradient-to-r from-purple-900/50 to-pink-900/50 border border-pink-500"
                  : "bg-zinc-900 border border-zinc-700 hover:border-zinc-500"
              }`}
              onClick={() => onSettingsChange("backgroundMusic", option.value)}
            >
              <div className={`p-2 rounded-full ${
                settings.backgroundMusic === option.value ? "bg-pink-500/30" : "bg-zinc-800"
              }`}>
                {option.value === "none" ? (
                  <VolumeX className={`h-5 w-5 ${
                    settings.backgroundMusic === option.value ? "text-pink-300" : "text-zinc-400"
                  }`} />
                ) : (
                  <Volume2 className={`h-5 w-5 ${
                    settings.backgroundMusic === option.value ? "text-pink-300" : "text-zinc-400"
                  }`} />
                )}
              </div>
              
              <div className="flex-1 ml-4">
                <div className="flex items-center">
                  <h3 className="text-lg font-medium text-white">{option.label}</h3>
                  {settings.backgroundMusic === option.value && (
                    <Check className="ml-2 h-4 w-4 text-pink-500" />
                  )}
                </div>
                <p className={`text-sm ${
                  settings.backgroundMusic === option.value ? "text-pink-200/70" : "text-zinc-400"
                }`}>
                  {option.description}
                </p>
              </div>
              
              {option.sample && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePlayback(option.value, option.sample);
                  }}
                  className={`ml-2 p-2 rounded-full ${
                    playingMusic === option.value 
                      ? "bg-pink-500/20 text-pink-400" 
                      : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                  }`}
                >
                  {playingMusic === option.value ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                </button>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}