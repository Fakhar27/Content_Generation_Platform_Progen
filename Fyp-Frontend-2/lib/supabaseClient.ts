import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL as string;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_SERVICE_KEY as string;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type Video = {
    id: string;
    user_id: string;
    filename: string;
    url: string;
    prompt: string;
    genre: string;
    created_at: string;
    metadata: {
      iterations: number;
      backgroundType: string;
      musicType: string;
      voiceType: string;
      subtitleColor: string;
    };
  };