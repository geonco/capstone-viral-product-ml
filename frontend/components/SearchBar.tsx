"use client";
import { useRouter } from "next/navigation";
import { useState, useMemo } from "react";
import { Search } from "lucide-react";

export function SearchBar({ keywords }: { keywords: string[] }) {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);

  const matches = useMemo(() => {
    if (!q.trim()) return [];
    return keywords.filter(k => k.includes(q.trim())).slice(0, 8);
  }, [q, keywords]);

  function go(kw: string) {
    router.push(`/keyword/${encodeURIComponent(kw)}`);
  }

  return (
    <div className="relative">
      <div className="flex items-center gap-3 card px-5 py-4">
        <Search size={20} className="text-sub" />
        <input
          value={q}
          onChange={(e) => { setQ(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && matches[0]) go(matches[0]);
          }}
          placeholder="키워드 검색 — 예: 약과쿠키, 두바이초콜릿, 베이글…"
          className="flex-1 bg-transparent outline-none text-base placeholder:text-sub"
        />
        {q && (
          <span className="text-xs text-sub">{matches.length}건</span>
        )}
      </div>
      {open && matches.length > 0 && (
        <div className="absolute left-0 right-0 mt-2 card overflow-hidden z-20">
          {matches.map((m) => (
            <button
              key={m}
              onMouseDown={() => go(m)}
              className="w-full text-left px-5 py-2.5 hover:bg-panel2 text-sm border-b border-border last:border-0"
            >
              {m}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
