"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { PlusCircle, Video, Image, FileText } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

export function CreateNewButton() {
  const router = useRouter()
  const [isHovered, setIsHovered] = useState(false)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          className="btn-gradient-pink relative group bg-pink-700"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <PlusCircle className="mr-2 h-4 w-4" />
          Create New
          <span
            className={`absolute inset-0 bg-black/10 rounded-md transition-opacity duration-300 ${
              isHovered ? "opacity-20" : "opacity-0"
            }`}
          />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="bg-zinc-950 border-zinc-800">
        <DropdownMenuItem
          className="flex items-center cursor-pointer hover:bg-zinc-900 text-white"
          onClick={() => router.push("/playground")}
        >
          <Video className="mr-2 h-4 w-4 text-pink-500" />
          <span>New Video</span>
        </DropdownMenuItem>
        <DropdownMenuItem className="flex items-center cursor-pointer hover:bg-zinc-900 text-white">
          <Image className="mr-2 h-4 w-4 text-cyan-500" />
          <span>New Image</span>
        </DropdownMenuItem>
        <DropdownMenuItem className="flex items-center cursor-pointer hover:bg-zinc-900 text-white">
          <FileText className="mr-2 h-4 w-4 text-green-500" />
          <span>New Script</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
