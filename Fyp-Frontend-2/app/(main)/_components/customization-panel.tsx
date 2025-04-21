"use client"

import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import type { VideoSettings } from "@/app/playground/page"
import { Step1Basic } from "./steps/step-1-basic"
import { Step2Background } from "./steps/step-2-basic"
import { Step3Music } from "./steps/step-3-basic"
import { Step4Voice } from "./steps/step-4-basic"
import { ArrowLeft, ArrowRight, Zap } from "lucide-react"

interface CustomizationPanelProps {
  settings: VideoSettings
  onSettingsChange: (key: keyof VideoSettings, value: any) => void
  currentStep: number
  onNextStep: () => void
  onPrevStep: () => void
}

export function CustomizationPanel({
  settings,
  onSettingsChange,
  currentStep,
  onNextStep,
  onPrevStep,
}: CustomizationPanelProps) {
  const steps = [
    { id: 1, name: "Basics" },
    { id: 2, name: "Background" },
    { id: 3, name: "Music" },
    { id: 4, name: "Voice" },
  ]

  return (
    <motion.div
      className="rounded-xl border border-zinc-800 bg-zinc-900 overflow-hidden"
      initial={{ opacity: 1 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 1 }}
    >
      <div className="border-b border-zinc-900 p-4">
        <h2 className="text-xl font-bold text-white">Customize Your Video</h2>
        <div className="flex mt-4">
          {steps.map((step) => (
            <div key={step.id} className="flex-1 relative">
              <div
                className={`h-2 ${
                  step.id <= currentStep ? "bg-gradient-to-r from-purple-600 to-pink-600" : "bg-zinc-800"
                } rounded-full`}
              ></div>
              <div
                className={`absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs ${
                  step.id <= currentStep ? "text-pink-500" : "text-zinc-600"
                }`}
              >
                {step.name}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="p-6 pt-10">
        <AnimatePresence mode="wait">
          {currentStep === 1 && <Step1Basic key="step1" settings={settings} onSettingsChange={onSettingsChange} />}
          {currentStep === 2 && <Step2Background key="step2" settings={settings} onSettingsChange={onSettingsChange} />}
          {currentStep === 3 && <Step3Music key="step3" settings={settings} onSettingsChange={onSettingsChange} />}
          {currentStep === 4 && <Step4Voice key="step4" settings={settings} onSettingsChange={onSettingsChange} />}
        </AnimatePresence>

        <div className="mt-8 flex justify-between">
          <Button
            variant="outline"
            onClick={onPrevStep}
            disabled={currentStep <= 1}
            className="border-zinc-700 text-zinc-400 hover:bg-zinc-800"
          >
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>
          <Button
            onClick={onNextStep}
            className={`${
              currentStep === 4
                ? "bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                : "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            } text-white font-bold`}
          >
            {currentStep === 4 ? (
              <>
                Generate <Zap className="ml-2 h-4 w-4" />
              </>
            ) : (
              <>
                Next <ArrowRight className="ml-2 h-4 w-4" />
              </>
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  )
}