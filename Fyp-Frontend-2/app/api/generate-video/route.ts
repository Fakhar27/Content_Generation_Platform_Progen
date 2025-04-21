// app/api/generate-video/route.ts
import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import { auth } from "@clerk/nextjs/server";
import { supabase } from '@/lib/supabaseClient';

// Get backend URL with robust fallback strategy
const BACKEND_URL = process.env.BACKEND_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'http://backend:8000'  // Docker service name for ECS internal communication
    : 'http://127.0.0.1:8000'); // Local development

export async function POST(request: NextRequest) {
  try {
    // Get the current authenticated user
    const { userId } = await auth();
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    const body = await request.json();

    console.log(`Attempting to connect to backend at ${BACKEND_URL}/generate-content/`);

    try {
      const response = await axios.post(`${BACKEND_URL}/generate-content/`, {
        prompt: body.prompt,
        genre: body.genre,
        iterations: body.iterations,
        backgroundType: body.backgroundType,
        musicType: body.musicType,
        voiceType: body.voiceType,
        subtitleColor: body.subtitleColor,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30 * 60 * 1000, // 30 minutes
      });

      const base64Video = response.data.video_data; 
      if (!base64Video) {
        return NextResponse.json({ error: 'No video data returned from backend' }, { status: 500 });
      }

      // Strip the data URL prefix if it's included
      const cleanedBase64 = base64Video.replace(/^data:video\/mp4;base64,/, '');

      // Convert base64 to a buffer
      const videoBuffer = Buffer.from(cleanedBase64, 'base64');
      
      const timestamp = new Date().getTime();
      const filename = `${userId}_${timestamp}.mp4`;
      
      const { data: uploadData, error: uploadError } = await supabase
        .storage
        .from('videos')
        .upload(`${userId}/${filename}`, videoBuffer, {
          contentType: 'video/mp4',
          cacheControl: '3600',
        });
      
      if (uploadError) {
        console.error('Supabase upload error:', uploadError);
        return NextResponse.json(
          { error: 'Failed to save video to storage', details: uploadError.message },
          { status: 500 }
        );
      }
    
      const { data: { publicUrl } } = supabase
        .storage
        .from('videos')
        .getPublicUrl(`${userId}/${filename}`);
      
      const { data: videoData, error: dbError } = await supabase
        .from('videos')
        .insert({
          user_id: userId,
          filename,
          url: publicUrl,
          prompt: body.prompt,
          genre: body.genre,
          created_at: new Date().toISOString(),
          metadata: {
            iterations: body.iterations,
            backgroundType: body.backgroundType,
            musicType: body.musicType,
            voiceType: body.voiceType,
            subtitleColor: body.subtitleColor,
          }
        })
        .select()
        .single();
      
      if (dbError) {
        console.error('Supabase database error:', dbError);
        return NextResponse.json(
          { error: 'Failed to save video metadata', details: dbError.message },
          { status: 500 }
        );
      }
      
      return NextResponse.json({
        success: true,
        videoId: videoData.id,
        url: publicUrl,
      });

    } catch (error: any) {
      console.error('Backend fetch error:', error);

      if (error.code === 'ECONNABORTED') {
        return NextResponse.json(
          {
            error: 'Request timed out after 30 minutes. Video generation is taking too long.',
          },
          { status: 504 }
        );
      }

      return NextResponse.json(
        {
          error: 'Failed to connect to video generation service',
          details: error.message || 'Unknown error',
        },
        { status: 502 }
      );
    }
  } catch (error: any) {
    console.error('API route error:', error);
    return NextResponse.json(
      {
        error: 'Error processing request',
        message: error.message,
      },
      { status: 500 }
    );
  }
}