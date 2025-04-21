"use client"

import { useState, Fragment } from "react";
import { Dialog, Transition } from "@headlessui/react";

interface DownloadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (filename: string) => void;
  defaultName: string;
}

export function DownloadModal({
  isOpen,
  onClose,
  onConfirm,
  defaultName,
}: DownloadModalProps) {
  const [filename, setFilename] = useState(defaultName);

  const handleConfirm = () => {
    if (!filename.trim()) return;
    onConfirm(filename);
    onClose();
  };

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/50" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-xl bg-zinc-900 p-6 text-left shadow-xl border border-zinc-700 transition-all">
                <Dialog.Title className="text-lg font-medium text-white">
                  Download Video
                </Dialog.Title>
                <p className="text-sm text-zinc-400 mt-2">
                  Enter a name for your video file:
                </p>

                <input
                  type="text"
                  value={filename}
                  onChange={(e) => setFilename(e.target.value)}
                  className="w-full mt-4 px-4 py-2 rounded bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter filename..."
                />

                <div className="mt-6 flex justify-end gap-3">
                  <button
                    onClick={onClose}
                    className="px-4 py-2 rounded bg-zinc-700 hover:bg-zinc-600 text-white"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleConfirm}
                    className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white font-semibold"
                  >
                    Confirm
                  </button>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}