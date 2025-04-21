"use client"

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { supabase } from "@/lib/supabaseClient";
import { useAuth } from "@clerk/nextjs";
import { Download, Play, Video, Trash2 } from "lucide-react";
import Link from "next/link";
import { AppSidebar } from "../(main)/_components/AppSidebar";

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

  const handleDelete = async (id: number, filename?: string) => {
    if (!userId || !filename) return;
    if (!confirm('Are you sure you want to delete this video?')) return;

    try {
      const { error: storageError } = await supabase
        .storage
        .from('videos')
        .remove([`${userId}/${filename}`]);

      const { error: dbError } = await supabase
        .from('videos')
        .delete()
        .eq('id', id);

      if (dbError) throw dbError;
      setVideos(videos.filter(video => video.id !== id));
    } catch (err: any) {
      console.error('Error deleting video:', err);
      alert('Failed to delete video: ' + err.message);
    }
  };

  return (
    <div className="flex min-h-screen w-full bg-black">
      <AppSidebar />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col w-full md:pl-64">
        <div className="relative">
          <div className="absolute inset-0 z-0 opacity-30">
            <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-pink-600/20 via-purple-600/10 to-transparent"></div>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,0,255,0.15),transparent_50%)]"></div>
          </div>
          <div className="relative w-full max-w-7xl mx-auto p-4 sm:p-6 z-10">
            <h1 className="text-3xl font-bold text-white mb-4">Exports</h1>
            <p className="text-zinc-400 mb-8">Your downloaded videos will appear here.</p>

            {loading ? (
              <div className="flex items-center justify-center py-20">
                <div className="w-8 h-8 border-4 border-zinc-500 border-t-zinc-300 rounded-full animate-spin"></div>
              </div>
            ) : videos.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-center text-zinc-400">
                <Video className="w-12 h-12 mb-4 text-zinc-500" />
                <p className="text-lg font-medium text-white">No downloaded videos found</p>
                <p className="text-sm mt-1 text-zinc-400">
                  Download some videos to see them here.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-zinc-800">
                {videos.map((video) => (
                  <motion.div
                    key={video.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 hover:bg-zinc-800/30 transition-colors"
                  >
                    <div className="flex flex-col sm:flex-row gap-4">
                      <div className="relative w-full sm:w-64 h-36 rounded overflow-hidden bg-zinc-900 flex-shrink-0">
                        <video
                          src={video.url}
                          className="w-full h-full object-cover"
                          poster="/video-placeholder.jpg"
                          muted
                          playsInline
                          autoPlay
                          loop
                          preload="auto"
                        />
                      </div>

                      <div className="flex-grow flex flex-col justify-between">
                        <div>
                          <Link href={video.url}>
                            <h3 className="font-medium truncate hover:text-blue-400 transition-colors">
                              {video.prompt.substring(0, 50)}{video.prompt.length > 50 ? '...' : ''}
                            </h3>
                          </Link>
                          <div className="flex items-center text-xs text-zinc-400 mt-1">
                            <span>{video.genre}</span>
                            <span className="mx-2">•</span>
                            <span>{formatDate(video.created_at)}</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 mt-4 sm:mt-0">
                          <Link
                            href={video.url}
                            className="p-1.5 rounded-full hover:bg-zinc-700 transition-colors"
                            title="View video"
                          >
                            <Play className="w-4 h-4" />
                          </Link>

                          <a
                            href={video.url}
                            download={video.filename}
                            className="p-1.5 rounded-full hover:bg-zinc-700 transition-colors"
                            title="Download video"
                          >
                            <Download className="w-4 h-4" />
                          </a>

                          <button
                            onClick={() => handleDelete(video.id, video.filename)}
                            className="p-1.5 rounded-full hover:bg-red-900/40 text-red-400 transition-colors"
                            title="Delete video"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
