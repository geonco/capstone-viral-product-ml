// 키워드 요약 카드 — 라벨 수치를 자연어 문장과 한줄 코멘트로 변환
// 첫 사용자가 표/차트 안 봐도 키워드 상태를 즉시 이해하게 함

type Matrix = Record<string, Record<string, number | null>>;
type Probs = Record<string, Record<string, number>>;
type Tone = "good" | "bad" | "neutral";

const TONE_BG: Record<Tone, string> = {
  good:    "from-emerald-500/[0.06] to-transparent border-emerald-500/20",
  bad:     "from-rose-500/[0.06] to-transparent border-rose-500/20",
  neutral: "from-slate-500/[0.04] to-transparent border-border",
};

const TONE_ICON: Record<Tone, string> = {
  good:    "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  bad:     "bg-rose-500/15 text-rose-300 border-rose-500/30",
  neutral: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

export function KeywordSummary({
  keyword, category, matrix, growthProbs, peakSoftpos15,
}: {
  keyword: string;
  category: string;
  matrix: Matrix;
  growthProbs: Probs;
  buzzProbs: Probs;
  peakSoftpos15: number | null;
}) {
  const g10 = matrix.growth?.["10"] ?? null;
  const s10 = matrix.sustainability?.["10"] ?? null;
  const b10 = matrix.buzz_composite?.["10"] ?? null;
  const c10 = matrix.crash?.["10"] ?? null;
  const sp10 = matrix.spike?.["10"] ?? null;

  type Line = { label: string; text: string; tone: Tone; icon: string };
  const lines: Line[] = [];

  if (g10 != null) {
    if (g10 >= 1.3)       lines.push({ label: "성장",   text: `향후 10일 검색량이 약 ${g10.toFixed(2)}배로 강하게 상승할 전망`, tone: "good", icon: "↑↑" });
    else if (g10 >= 1.05) lines.push({ label: "성장",   text: `향후 10일 검색량이 약 ${g10.toFixed(2)}배로 완만하게 상승`, tone: "good", icon: "↑" });
    else if (g10 >= 0.95) lines.push({ label: "성장",   text: `향후 10일 검색량은 현재 수준 유지 (${g10.toFixed(2)}배)`, tone: "neutral", icon: "→" });
    else if (g10 >= 0.75) lines.push({ label: "성장",   text: `향후 10일 검색량이 약 ${g10.toFixed(2)}배로 다소 감소`, tone: "bad", icon: "↓" });
    else                  lines.push({ label: "성장",   text: `향후 10일 검색량이 약 ${g10.toFixed(2)}배로 큰 폭 감소 전망`, tone: "bad", icon: "↓↓" });
  }

  if (s10 != null) {
    if (s10 >= 1.5)       lines.push({ label: "지속성", text: `유행 지속성이 매우 높음 — 현재 흐름이 오래 갈 가능성`, tone: "good", icon: "◆" });
    else if (s10 >= 1.1)  lines.push({ label: "지속성", text: `유행 지속성 양호`, tone: "good", icon: "◆" });
    else if (s10 >= 0.8)  lines.push({ label: "지속성", text: `유행 지속성 보통`, tone: "neutral", icon: "◇" });
    else                  lines.push({ label: "지속성", text: `유행이 단발성 — 빠르게 식을 가능성`, tone: "bad", icon: "◇" });
  }

  if (b10 != null) {
    if (b10 >= 1.5)        lines.push({ label: "버즈",   text: `버즈가 평균 대비 매우 강함 (z=${b10.toFixed(2)})`, tone: "good", icon: "✦" });
    else if (b10 >= 0.3)   lines.push({ label: "버즈",   text: `버즈가 평균 이상 (z=${b10.toFixed(2)})`, tone: "good", icon: "✦" });
    else if (b10 >= -0.3)  lines.push({ label: "버즈",   text: `버즈는 평균 수준 (z=${b10.toFixed(2)})`, tone: "neutral", icon: "·" });
    else                   lines.push({ label: "버즈",   text: `버즈가 평균 이하 — 화제성 약함 (z=${b10.toFixed(2)})`, tone: "bad", icon: "·" });
  }

  if (sp10 != null && sp10 >= 0.5) {
    lines.push({ label: "급등", text: `급등 가능성 ${(sp10 * 100).toFixed(0)}% — 단기 스파이크 주의`, tone: "good", icon: "▲" });
  }
  if (c10 != null && c10 >= 0.3) {
    lines.push({ label: "하락", text: `하락 위험 ${(c10 * 100).toFixed(0)}% — 최고점 대비 큰 폭 조정 가능`, tone: "bad", icon: "▼" });
  }

  if (peakSoftpos15 != null) {
    const day = Math.round(peakSoftpos15);
    if (day <= 3)        lines.push({ label: "피크",   text: `피크 시점이 매우 가까움 — 약 ${day}일 이내`, tone: "neutral", icon: "◉" });
    else if (day <= 8)   lines.push({ label: "피크",   text: `피크 시점은 약 ${day}일 후`, tone: "neutral", icon: "◎" });
    else                 lines.push({ label: "피크",   text: `피크 시점은 약 ${day}일 후 — 천천히 달아오를 가능성`, tone: "neutral", icon: "○" });
  }

  const gp = growthProbs.growth_10d;
  if (gp) {
    const best = Object.entries(gp).sort((a, b) => b[1] - a[1])[0];
    if (best && best[1] >= 0.5) {
      const labelMap: Record<string, string> = {
        shrink: "축소", stable: "유지", surge: "급증", extreme: "폭발적",
      };
      lines.push({
        label: "구간",
        text: `성장 구간 분류: '${labelMap[best[0]] ?? best[0]}' 가능성 ${(best[1] * 100).toFixed(0)}%`,
        tone: best[0] === "extreme" || best[0] === "surge" ? "good" : best[0] === "shrink" ? "bad" : "neutral",
        icon: "◐",
      });
    }
  }

  const goodCount = lines.filter(l => l.tone === "good").length;
  const badCount  = lines.filter(l => l.tone === "bad").length;
  let headline: string;
  let headlineTone: Tone;
  if (goodCount >= badCount + 2) {
    headline = `${category} 카테고리에서 상승 모멘텀이 뚜렷한 키워드`;
    headlineTone = "good";
  } else if (badCount >= goodCount + 2) {
    headline = `단기 약세 신호가 우세 — 진입에 신중이 필요`;
    headlineTone = "bad";
  } else if (goodCount > badCount) {
    headline = `일부 긍정 신호가 있으나 혼조 — 관찰 권장`;
    headlineTone = "neutral";
  } else if (badCount > goodCount) {
    headline = `일부 부정 신호가 있으나 혼조 — 관찰 권장`;
    headlineTone = "neutral";
  } else {
    headline = `평이한 흐름 — 뚜렷한 방향성 없음`;
    headlineTone = "neutral";
  }

  const headlineBadge: Record<Tone, { dot: string; ring: string; chip: string }> = {
    good:    { dot: "bg-emerald-400", ring: "ring-emerald-500/20", chip: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30" },
    bad:     { dot: "bg-rose-400",    ring: "ring-rose-500/20",    chip: "bg-rose-500/10 text-rose-300 border-rose-500/30" },
    neutral: { dot: "bg-slate-400",   ring: "ring-slate-500/20",   chip: "bg-slate-500/10 text-slate-300 border-slate-500/30" },
  };
  const hb = headlineBadge[headlineTone];

  return (
    <div className={`relative overflow-hidden rounded-2xl border bg-gradient-to-br ${TONE_BG[headlineTone]} backdrop-blur-sm`}>
      {/* 우상단 장식 글로우 */}
      <div className="pointer-events-none absolute -top-20 -right-20 h-56 w-56 rounded-full bg-gradient-to-br from-accent/20 to-accent2/10 blur-3xl" />

      <div className="relative p-6">
        <div className="flex items-center justify-between gap-4 mb-5">
          <div className="flex items-center gap-2.5">
            <span className={`relative inline-flex h-2 w-2`}>
              <span className={`absolute inline-flex h-full w-full rounded-full ${hb.dot} opacity-60 animate-ping`} />
              <span className={`relative inline-flex h-2 w-2 rounded-full ${hb.dot}`} />
            </span>
            <h2 className="text-sm font-semibold tracking-wide uppercase text-sub">한눈에 요약</h2>
          </div>
          <span className={`text-[11px] px-2 py-1 rounded-full border ${hb.chip} font-medium`}>
            AI 모델 종합 판단
          </span>
        </div>

        <p className="text-xl md:text-2xl font-bold tracking-tight text-text mb-1 leading-snug">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent to-accent2">{keyword}</span>
          <span className="text-text">은(는) {headline}</span>
        </p>
        <p className="text-xs text-sub mb-6">라벨 값을 자연어로 해석한 요약 — 아래 차트·표와 함께 보면 더 정확</p>

        <ul className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {lines.map((l, i) => (
            <li
              key={i}
              className={`group flex items-start gap-3 p-3 rounded-xl border bg-panel2/40 hover:bg-panel2/70 transition-colors ${
                l.tone === "good" ? "border-emerald-500/15" :
                l.tone === "bad"  ? "border-rose-500/15" :
                "border-border"
              }`}
            >
              <span className={`inline-flex items-center justify-center w-7 h-7 rounded-lg border text-[11px] font-bold shrink-0 ${TONE_ICON[l.tone]}`}>
                {l.icon}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-[10px] uppercase tracking-wider text-sub mb-0.5 font-semibold">{l.label}</div>
                <div className="text-sm text-text leading-snug">{l.text}</div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
