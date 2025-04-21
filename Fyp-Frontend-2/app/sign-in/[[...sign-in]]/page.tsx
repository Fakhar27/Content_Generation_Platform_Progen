import { ClerkLoaded, ClerkProvider, SignIn } from "@clerk/nextjs"

export default function SignInPage() {
  return (
    <div className="flex justify-center w-full bg-black">
      <div className="w-full max-w-md p-6">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-white mb-2">Sign In</h1>
          <p className="text-zinc-400">Sign in to access your AI video generator dashboard</p>
        </div>
        <ClerkProvider>
        <SignIn
          appearance={{
            elements: {
              formButtonPrimary: "bg-pink-600 hover:bg-pink-700 text-sm normal-case",
              card: "bg-zing-600 border border-zinc-800",
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
          path="/sign-in"
          signUpUrl="/sign-up"
          redirectUrl="/dashboard"
        />
        </ClerkProvider>
      </div>
    </div>
  )
}
