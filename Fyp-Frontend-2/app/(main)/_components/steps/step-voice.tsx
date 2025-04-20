"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Mic, Play, Pause, Volume2, Check } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

interface StepVoiceProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepVoice({ settings, onSettingsChange }: StepVoiceProps) {
  const [playingVoice, setPlayingVoice] = useState<string | null>(null);
  const [voiceVolume, setVoiceVolume] = useState(80);
  const [voiceSpeed, setVoiceSpeed] = useState(100);
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

  const voiceOptions = [
    {
      id: "en_speaker_4",
      name: "William",
      description: "Mature male voice, perfect for documentary-style videos",
      tags: ["Male", "Mature", "English"],
      sample: "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_4.mp3",
    },
    {
      id: "en_speaker_6",
      name: "Adam",
      description: "Energetic male voice, popular for TikTok and short videos",
      tags: ["Male", "Middle aged", "English"],
      sample: "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_6.mp3",
    },
    {
      id: "en_speaker_9",
      name: "Natasha",
      description: "Soft female voice, great for informative and calming content",
      tags: ["Female", "Middle aged", "English"],
      sample: "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_9.mp3",
    },
    {
      id: "zh_speaker_8",
      name: "Jake",
      description: "Gentle whisper voice, ideal for ASMR and relaxation videos",
      tags: ["Male", "Young", "English"],
      sample: "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_8.mp3",
    },
  ];

  // Audio setup is now handled in the main useEffect

  const togglePlayback = (voiceId: string, sampleUrl: string) => {
    try {
      if (playingVoice === voiceId) {
        if (audioRef.current) {
          audioRef.current.pause();
        }
        setPlayingVoice(null);
      } else {
        if (!audioRef.current) {
          audioRef.current = new Audio(sampleUrl);
        } else {
          audioRef.current.src = sampleUrl;
        }
        
        // Apply settings
        audioRef.current.volume = voiceVolume / 100;
        audioRef.current.playbackRate = voiceSpeed / 100;
        
        const playPromise = audioRef.current.play();
        
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              setPlayingVoice(voiceId);
            })
            .catch(err => {
              console.error("Failed to play audio:", err);
              // Try alternative playback method for some browsers
              setTimeout(() => {
                if (audioRef.current) {
                  audioRef.current.play()
                    .then(() => setPlayingVoice(voiceId))
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
      <div className="flex flex-col space-y-6">
        <p className="text-zinc-400 text-sm mb-2">
          Choose a voice for your video narration. Each voice has unique characteristics that can match different content styles.
        </p>
        
        <audio 
          ref={audioRef} 
          className="hidden" 
          onEnded={() => setPlayingVoice(null)}
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {voiceOptions.map((voice) => (
            <motion.div
              key={voice.id}
              whileHover={{ scale: 1.03 }}
              className={`flex flex-col rounded-lg border overflow-hidden ${
                settings.voiceType === voice.id
                  ? "border-pink-500 bg-gradient-to-br from-zinc-900 to-pink-950/20"
                  : "border-zinc-700 bg-zinc-900 hover:border-zinc-500"
              }`}
            >
              <div 
                className="p-4 cursor-pointer"
                onClick={() => onSettingsChange("voiceType", voice.id)}
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-lg font-medium text-white flex items-center">
                      {voice.name}
                      {settings.voiceType === voice.id && (
                        <Check className="ml-2 h-4 w-4 text-pink-500" />
                      )}
                    </h3>
                    <p className="text-sm text-zinc-400">{voice.description}</p>
                  </div>
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      togglePlayback(voice.id, voice.sample);
                    }}
                    className={`p-2 rounded-full ${
                      playingVoice === voice.id 
                        ? "bg-pink-500/20 text-pink-400" 
                        : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                    }`}
                  >
                    {playingVoice === voice.id ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </button>
                </div>
                
                <div className="flex flex-wrap gap-2 mt-2">
                  {voice.tags.map((tag) => (
                    <span 
                      key={tag} 
                      className="px-2 py-1 text-xs rounded-full bg-zinc-800 text-zinc-300"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                
                {playingVoice === voice.id && (
                  <div className="mt-3 h-1 bg-zinc-800 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-pink-500 to-purple-500"
                      initial={{ width: "0%" }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 3, ease: "linear" }}
                    />
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
        
        
      </div>
    </div>
  );
}