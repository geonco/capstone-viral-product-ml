import clsx from "clsx";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";

type Tone = "neutral" | "good" | "bad" | "warn";

export function KpiCard({
  label,
  value,
  unit,
  delta,
  tone = "neutral",
  hint,
}: {
  label: string;
  value: string | number;
  unit?: string;
  delta?: number;
  tone?: Tone;
  hint?: string;
}) {
  const toneRing = {
    neutral: "ring-border",
    good: "ring-good/30",
    bad: "ring-bad/30",
    warn: "ring-warn/30",
  }[tone];
  const dot = {
    neutral: "bg-sub",
    good: "bg-good",
    bad: "bg-bad",
    warn: "bg-warn",
  }[tone];
  const Icon = delta == null ? Minus : delta > 0 ? ArrowUp : ArrowDown;
  const deltaCls = delta == null ? "text-sub" : delta > 0 ? "text-good" : "text-bad";
  return (
    <div className={clsx("card p-4 ring-1", toneRing)}>
      <div className="flex items-center gap-1.5 text-xs text-sub">
        <span className={clsx("w-1.5 h-1.5 rounded-full", dot)} />
        {label}
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className="text-3xl font-bold tracking-tight">{value}</span>
        {unit && <span className="text-sm text-sub">{unit}</span>}
      </div>
      <div className="mt-1 flex items-center gap-2 text-xs">
        {delta != null && (
          <span className={clsx("inline-flex items-center gap-0.5", deltaCls)}>
            <Icon size={12} />
            {Math.abs(delta).toFixed(1)}%
          </span>
        )}
        {hint && <span className="text-sub">{hint}</span>}
      </div>
    </div>
  );
}
