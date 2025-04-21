import { supabase } from "@/lib/supabaseClient";
import { NextResponse } from "next/server";

export async function GET() {
    const { data, error } = await supabase.storage.from("videos").list();

    if (error) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }

    // Corrected way to get public URLs
    const videoUrls = data.map((file) => ({
        name: file.name,
        url: supabase.storage.from("fypcontent").getPublicUrl(file.name).data.publicUrl, // ✅ Corrected
    }));

    return NextResponse.json({ videos: videoUrls }, { status: 200 });
}