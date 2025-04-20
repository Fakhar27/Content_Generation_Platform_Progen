"use client";

import { motion } from "framer-motion";
import { useRef, useState, useEffect } from "react";
import { Play, Pause, CheckCircle } from "lucide-react";

interface StepBackgroundVideoProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onPromptSubmit: (value: string) => void;
  settings: any;
  onSettingsChange: (key: string, value: any) => void;
}

export function StepBackgroundVideo({ settings, onSettingsChange }: StepBackgroundVideoProps) {
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);
  const videoRefs = useRef<{ [key: string]: HTMLVideoElement | null }>({});

  const backgroundOptions = [
    {
      value: "1",
      label: "",
      description: "Minecraft gameplay",
      preview: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/fypcontent//MC%20Parkour%20(3)%20(1)%20(1)%20(1)%20(1).mp4",
    },
    {
      value: "2",
      label: "",
      description: "Subway Surfer gameplay",
      preview: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/fypcontent//SubwaySurfer.mp4",
    },
    {
      value: "3",
      label: "",
      description: "Natural scenery with flowing water and trees",
      preview: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/fypcontent//backgroundVideo69.mp4", 
    },
    {
      value: "4",
      label: "",
      description: "Clean and simple design with neutral tones",
      preview: "https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/fypcontent//bro.mp4",
    },
    {
      value: "0",
      label: "No Video",
      description: "Generate video without background Video",
      preview: null,
    },
  ];

  useEffect(() => {
    // Cleanup function to pause all videos when component unmounts
    return () => {
      Object.values(videoRefs.current).forEach(videoRef => {
        if (videoRef) videoRef.pause();
      });
    };
  }, []);

  const togglePlayback = (videoId: string) => {
    const videoRef = videoRefs.current[videoId];
    
    if (!videoRef) return;
    
    try {
      if (playingVideo === videoId) {
        videoRef.pause();
        setPlayingVideo(null);
      } else {
        // Pause any currently playing video
        if (playingVideo && videoRefs.current[playingVideo]) {
          videoRefs.current[playingVideo]?.pause();
        }
        
        const playPromise = videoRef.play();
        
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              setPlayingVideo(videoId);
            })
            .catch(error => {
              console.error("Error playing video:", error);
              // Try with a slight delay for some browsers
              setTimeout(() => {
                videoRef.play()
                  .then(() => setPlayingVideo(videoId))
                  .catch(e => console.error("Second attempt failed:", e));
              }, 100);
            });
        }
      }
    } catch (error) {
      console.error("Video playback error:", error);
    }
  };

  return (
    <div className="max-w-xl mx-auto">
      <div className="flex flex-col space-y-4">
        <p className="text-zinc-400 text-sm mb-2">
          Select a background style for your video. The background will set the overall mood and theme.
        </p>
        
        <div className="grid grid-cols-1 gap-4">
          {backgroundOptions.map((option) => (
            <motion.div
              key={option.value}
              whileHover={{ scale: 1.02 }}
              className={`relative rounded-lg cursor-pointer transition-colors border overflow-hidden ${
                settings.backgroundVideo === option.value
                  ? "border-pink-500"
                  : "border-zinc-700 hover:border-zinc-500"
              }`}
              onClick={() => onSettingsChange("backgroundVideo", option.value)}
            >
              <div className="relative">
                <video
                  ref={el => videoRefs.current[option.value] = el}
                  src={option.preview}
                  className="w-full h-40 object-cover"
                  loop
                  autoPlay
                  muted
                  playsInline
                  onEnded={() => setPlayingVideo(null)}
                />
                
                {/* Play/pause button */}
                <button
                  className="absolute bottom-3 right-3 p-2 rounded-full bg-black/50 hover:bg-black/70 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePlayback(option.value);
                  }}
                >
                  {playingVideo === option.value ? (
                    <Pause className="w-4 h-4 text-white" />
                  ) : (
                    <Play className="w-4 h-4 text-white" />
                  )}
                </button>
                
                {/* Selection indicator */}
                {settings.backgroundVideo === option.value && (
                  <div className="absolute top-3 right-3">
                    <CheckCircle className="w-6 h-6 text-pink-500" />
                  </div>
                )}
              </div>
              
              <div className="p-4 bg-zinc-900">
                <h3 className="text-lg font-medium text-white">{option.label}</h3>
                <p className="text-sm text-zinc-400">{option.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}