// lib/supabaseClient.ts
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_SERVICE_KEY;

// Validate required environment variables
if (!supabaseUrl || !supabaseAnonKey) {
  console.error(
    "Missing Supabase environment variables. Please make sure NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_SERVICE_KEY are set."
  );
  
  // In production, we'll throw an error; in development, we'll use dummy values
  if (process.env.NODE_ENV === 'production') {
    throw new Error("Missing required Supabase environment variables");
  }
}

export const supabase = createClient(
  supabaseUrl as string,
  supabaseAnonKey as string
);

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

// import { createClient } from "@supabase/supabase-js";

// const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL as string;
// const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_SERVICE_KEY as string;

// export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// export type Video = {
//     id: string;
//     user_id: string;
//     filename: string;
//     url: string;
//     prompt: string;
//     genre: string;
//     created_at: string;
//     metadata: {
//       iterations: number;
//       backgroundType: string;
//       musicType: string;
//       voiceType: string;
//       subtitleColor: string;
//     };
//   };