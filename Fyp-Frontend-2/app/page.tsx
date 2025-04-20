"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import {
  Video,
  Sparkles,
  Layers,
  Wand2,
  Share2,
  ChevronRight,
  CheckCircle2,
  Github,
  Twitter,
  Instagram,
} from "lucide-react"
import { useAuth, UserButton, SignedIn, SignedOut } from "@clerk/nextjs";


export default function LandingPage() {
  
  const [isScrolled, setIsScrolled] = useState(false)

  
  const { isSignedIn, userId } = useAuth();
  
 useEffect(() => {
      const handleScroll = () => {
        setIsScrolled(window.scrollY > 50);
      };
      window.addEventListener("scroll", handleScroll);
      return () => window.removeEventListener("scroll", handleScroll);
    }, []);
  


  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 },
    },
  }

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  }

  const features = [
    {
      title: "AI-Powered Generation",
      description:
        "Create stunning videos with just a text prompt. Our AI understands your vision and brings it to life.",
      icon: Sparkles,
      color: "pink",
    },
    {
      title: "Customizable Styles",
      description: "Choose from various punk-inspired styles, from cyberpunk to retro-punk and everything in between.",
      icon: Layers,
      color: "cyan",
    },
    {
      title: "Advanced Editing",
      description: "Fine-tune your videos with our intuitive editing tools. Adjust colors, effects, and more.",
      icon: Wand2,
      color: "purple",
    },
    {
      title: "Easy Sharing",
      description: "Share your creations directly to social media or download in multiple formats.",
      icon: Share2,
      color: "green",
    },
  ]

  const steps = [
    {
      number: "01",
      title: "Enter Your Prompt",
      description: "Describe your video idea in detail. The more specific you are, the better the results.",
    },
    {
      number: "02",
      title: "Customize Settings",
      description: "Choose your preferred style, duration, and other settings to perfect your vision.",
    },
    {
      number: "03",
      title: "Generate & Edit",
      description: "Our AI generates your video, then use our tools to make any final adjustments.",
    },
    {
      number: "04",
      title: "Share Your Creation",
      description: "Download your video or share it directly to your favorite platforms.",
    },
  ]

  const plans = [
    {
      name: "Free",
      price: "$0",
      description: "Perfect for trying out the platform",
      features: ["5 video generations per month", "720p resolution", "Basic editing tools", "Standard rendering speed"],
      cta: "Get Started",
      highlighted: false,
    },
    {
      name: "Pro",
      price: "$19",
      period: "/month",
      description: "For creators who need more power",
      features: [
        "50 video generations per month",
        "1080p resolution",
        "Advanced editing tools",
        "Priority rendering",
        "No watermark",
      ],
      cta: "Go Pro",
      highlighted: true,
    },
    {
      name: "Enterprise",
      price: "Custom",
      description: "For teams and businesses",
      features: [
        "Unlimited video generations",
        "4K resolution",
        "All editing features",
        "Fastest rendering",
        "API access",
        "Dedicated support",
      ],
      cta: "Contact Us",
      highlighted: false,
    },
  ]

  return (
    <div className="flex flex-col min-h-screen bg-black">
      {/* Header */}
      <header
      className={`fixed w-full z-50 transition-all duration-300 ${
        isScrolled ? "bg-black/80 backdrop-blur-md shadow-md" : "bg-transparent"
      }`}
    >
      <div className="container mx-auto px-4 lg:px-6 h-16 flex items-center">
        {/* Logo */}
        <Link href="/" className="flex items-center">
          <div className="w-8 h-8 rounded-md bg-gradient-to-br from-pink-600 to-purple-600 flex items-center justify-center mr-2">
            <Video className="h-4 w-4 text-white" />
          </div>
          <span className="text-xl font-bold text-white">PROGEN.AI</span>
        </Link>

        {/* Navigation Links */}
        <nav className="hidden md:flex ml-10 space-x-8">
        </nav>

        {/* Right Side - Auth & Actions */}
        <div className="ml-auto flex gap-4 sm:gap-6">
         <Link href="/dashboard">
            <Button variant="outline" size="sm" className="border-pink-600 text-pink-500 hover:bg-pink-600/10">
              Dashboard
            </Button>
          </Link>
          <Link href="/playground">
            <Button variant="outline" size="sm" className="border-pink-600 text-pink-500 hover:bg-pink-600/10">
              Try Playground
            </Button>
          </Link>

          {/* Authentication Section */}
          <SignedOut>
            <Link href="/sign-in">
              <Button variant="outline" size="sm" className="border-blue-600 text-blue-500 hover:bg-blue-600/10">
                Sign In
              </Button>
            </Link>
          </SignedOut>

          <SignedIn>
            <UserButton afterSignOutUrl="/" />
          </SignedIn>
        </div>
      </div>
    </header>

      {/* Hero Section */}
      <section className="relative pt-24 pb-20 md:pt-32 md:pb-28 lg:pt-40 lg:pb-36 overflow-hidden">
        <div className="absolute inset-0 z-0 opacity-30">
          <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-pink-600/20 via-purple-600/10 to-transparent"></div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,0,255,0.15),transparent_50%)]"></div>
        </div>

        <div className="container mx-auto px-4 relative z-10">
          <motion.div className="max-w-4xl mx-auto text-center" initial="hidden" animate="visible" variants={fadeIn}>
            <motion.h1
              className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tighter text-white mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              Generate Stunning
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-500">
                {" "}
                AI Videos{" "}
              </span>
            </motion.h1>

            <motion.p
              className="text-zinc-400 text-lg md:text-xl max-w-2xl mx-auto mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              Create professional-quality videos with just a text prompt. Our AI understands your vision and brings it
              to life with a unique aesthetic.
            </motion.p>

            <motion.div
              className="flex flex-col sm:flex-row gap-4 justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <Link href="/dashboard">
                <Button className="bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 text-white px-8 py-6 h-auto text-lg">
                  Get Started
                </Button>
              </Link>
            </motion.div>
          </motion.div>

          <motion.div
            className="mt-16 relative max-w-5xl mx-auto"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >  
             <div className="aspect-video rounded-xl overflow-hidden border border-zinc-800 shadow-2xl shadow-pink-500/10 group relative">
  <video
    src="https://qbhnvyynbrkwonkfxkwn.supabase.co/storage/v1/object/public/fypcontent//WEbsite%20Demo%20(1).mp4"
    autoPlay
    loop
    playsInline
    className="w-full h-full object-cover rounded-xl"
  />
  <div className="absolute inset-0 transition-all duration-300 group-hover:bg-black/10 pointer-events-none rounded-xl" />
</div>

            <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-gradient-to-br from-pink-600 to-purple-600 rounded-full blur-3xl opacity-30"></div>
            <div className="absolute -top-6 -left-6 w-32 h-32 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-full blur-3xl opacity-20"></div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 md:py-28 bg-zinc-950">
        <div className="container mx-auto px-4">
          <motion.div
            className="text-center mb-16"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeIn}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Powerful Features</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              Our AI video generator comes packed with features to help you create stunning videos in minutes.
            </p>
          </motion.div>

          <motion.div
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-8"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 hover:border-pink-500/50 transition-colors"
                variants={fadeIn}
              >
                <div
                  className={`w-12 h-12 rounded-lg bg-${feature.color}-500/20 flex items-center justify-center mb-4`}
                >
                  <feature.icon className={`h-6 w-6 text-${feature.color}-500`} />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-zinc-400">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 md:py-28">
        <div className="container mx-auto px-4">
          <motion.div
            className="text-center mb-16"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeIn}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">How It Works</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              Creating stunning AI videos is easy with our simple four-step process.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-16 items-center">
            <motion.div
              className="relative"
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.6 }}
            >
              <div className="aspect-[4/5] rounded-xl overflow-hidden border border-zinc-800 shadow-xl">
                <Image
                  src="/Chinatown.jpg"
                  alt="AI Video Generation Process"
                  width={640}
                  height={800}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-gradient-to-br from-pink-600 to-purple-600 rounded-full blur-3xl opacity-30"></div>
            </motion.div>

            <motion.div
              className="space-y-8"
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={staggerContainer}
            >
              {steps.map((step, index) => (
                <motion.div key={index} className="flex gap-4" variants={fadeIn}>
                  <div className="flex-shrink-0 w-12 h-12 rounded-full bg-pink-500/20 flex items-center justify-center text-pink-500 font-bold">
                    {step.number}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-2">{step.title}</h3>
                    <p className="text-zinc-400">{step.description}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 md:py-28 bg-gradient-to-b from-zinc-950 to-black">
        <div className="container mx-auto px-4">
          <motion.div
            className="max-w-4xl mx-auto text-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Ready to Create Amazing Videos?</h2>
            <p className="text-zinc-400 text-lg mb-8 max-w-2xl mx-auto">
              Join thousands of creators who are already using Progen.AI to bring their video ideas to life.
            </p>
            <Link href="/dashboard">
              <Button className="bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 text-white px-8 py-6 h-auto text-lg">
                Start Creating Now
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black border-t border-zinc-800 pt-16 pb-8">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-8 mb-12">
            <div className="col-span-2 lg:col-span-2">
              <Link href="/" className="flex items-center mb-4">
                <div className="w-8 h-8 rounded-md bg-gradient-to-br from-pink-600 to-purple-600 flex items-center justify-center mr-2">
                  <Video className="h-4 w-4 text-white" />
                </div>
                <span className="text-xl font-bold text-white">PROGEN.AI</span>
              </Link>
              <p className="text-zinc-400 mb-4 max-w-xs">
                Create stunning AI-generated videos with a aesthetic. Customize your videos with our powerful
                tools.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-zinc-400 hover:text-white transition-colors">
                  <Twitter className="h-5 w-5" />
                </a>
                <a href="#" className="text-zinc-400 hover:text-white transition-colors">
                  <Instagram className="h-5 w-5" />
                </a>
                <a href="#" className="text-zinc-400 hover:text-white transition-colors">
                  <Github className="h-5 w-5" />
                </a>
              </div>
            </div>

            <div>
              <h3 className="text-white font-semibold mb-4">Product</h3>
              <ul className="space-y-2">
                <li>
                  <Link href="#features" className="text-zinc-400 hover:text-white transition-colors">
                    Features
                  </Link>
                </li>
                <li>
                  <Link href="#pricing" className="text-zinc-400 hover:text-white transition-colors">
                    Pricing
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Roadmap
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Changelog
                  </Link>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-white font-semibold mb-4">Resources</h3>
              <ul className="space-y-2">
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Documentation
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Tutorials
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Blog
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Support
                  </Link>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-white font-semibold mb-4">Company</h3>
              <ul className="space-y-2">
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    About
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Careers
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Privacy
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-zinc-400 hover:text-white transition-colors">
                    Terms
                  </Link>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-zinc-800 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-zinc-500 text-sm mb-4 md:mb-0">© 2024 PROGEN.AI. All rights reserved.</p>
            <div className="flex space-x-6">
              <Link href="#" className="text-zinc-500 hover:text-white text-sm transition-colors">
                Privacy Policy
              </Link>
              <Link href="#" className="text-zinc-500 hover:text-white text-sm transition-colors">
                Terms of Service
              </Link>
              <Link href="#" className="text-zinc-500 hover:text-white text-sm transition-colors">
                Cookie Policy
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
