import { ClerkProvider, SignUp } from "@clerk/nextjs"

export default function SignUpPage() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="w-full max-w-md p-6">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-white mb-2">Create Account</h1>
          <p className="text-zinc-400">Sign up to start creating AI videos</p>
        </div>
        <ClerkProvider>
        <SignUp

          appearance={{
            elements: {
              formButtonPrimary: "bg-pink-600 hover:bg-pink-700 text-sm normal-case",
              card: "bg-zinc-900 border border-zinc-800",
              headerTitle: "text-white",
              headerSubtitle: "text-zinc-400",
              formFieldLabel: "text-zinc-300",
              formFieldInput: "bg-zinc-800 border-zinc-700 text-white",
              footerActionLink: "text-pink-500 hover:text-pink-400",
              identityPreviewText: "text-white",
              identityPreviewEditButton: "text-pink-500 hover:text-pink-400",
            },
          }}
          routing="path"
          path="/sign-up"
          signInUrl="/sign-in"
          redirectUrl="/dashboard"
        />
        </ClerkProvider>
      </div>
    </div>
  )
}