"use client"

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { supabase } from "@/lib/supabaseClient";
import { useAuth } from "@clerk/nextjs";
import { Download, Play, Video } from "lucide-react";
import Link from "next/link";
import SidebarWrapper from "../(main)/_components/SidebarWrapper";

interface ExportedVideo {
  id: number;
  url: string;
  prompt: string;
  genre: string;
  created_at: string;
  filename: string;
}

export default function ExportsPage() {
  const { userId } = useAuth();
  const [videos, setVideos] = useState<ExportedVideo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchDownloadedVideos() {
      if (!userId) return;

      setLoading(true);
      const { data, error } = await supabase
        .from("videos")
        .select("*")
        .eq("user_id", userId)
        .eq("downloaded", true)
        .order("created_at", { ascending: false });

      if (!error) setVideos(data || []);
      setLoading(false);
    }

    fetchDownloadedVideos();
  }, [userId]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="flex min-h-screen w-full">
    <div className="w-64">
         <SidebarWrapper />
    </div>
    <div className="flex min-h-screen w-full bg-zinc-900">
      <div className="flex-1 max-w-6xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-white mb-6">Exported Videos</h1>

        {loading ? (
          <div className="text-center text-zinc-400">Loading your videos...</div>
        ) : videos.length === 0 ? (
          <div className="text-center py-20 text-zinc-400">
            <Video className="mx-auto w-12 h-12 mb-4" />
            <p className="text-lg font-medium text-white">No exported videos yet</p>
            <p className="text-sm mt-1">Download a video from your dashboard to view it here.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {videos.map((video, index) => (
              <motion.div
                key={video.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700"
              >
                <video
                  src={video.url}
                  className="w-full h-48 object-cover"
                  muted
                  playsInline
                  preload="metadata"
                  controls
                  poster="/video-placeholder.jpg"
                />
                <div className="p-4">
                  <h2 className="text-white font-medium text-lg truncate">
                    {video.prompt.substring(0, 50)}{video.prompt.length > 50 ? "..." : ""}
                  </h2>
                  <p className="text-sm text-zinc-400 mt-1">{video.genre} • {formatDate(video.created_at)}</p>
                  <div className="flex gap-3 mt-4">
                    <Link href={video.url} target="_blank" className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1">
                      <Play className="w-4 h-4" /> View
                    </Link>
                    <a
                      href={video.url}
                      download={video.filename}
                      className="text-green-400 hover:text-green-300 text-sm flex items-center gap-1"
                    >
                      <Download className="w-4 h-4" /> Download
                    </a>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
    </div>
  );
}
