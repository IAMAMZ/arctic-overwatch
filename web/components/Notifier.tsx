"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

export default function Notifier({ message }: { message: string | null }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!message) return setVisible(false);
    setVisible(true);
    const t = setTimeout(() => setVisible(false), 5000); // auto-hide
    return () => clearTimeout(t);
  }, [message]);

  if (!visible || !message) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[100]">
      <Link
        href="/map"
        className="block rounded-xl bg-black/80 px-4 py-3 text-sm text-white shadow-lg backdrop-blur-md transition hover:bg-black/90"
      >
        🚢 {message}
      </Link>
    </div>
  );
}
