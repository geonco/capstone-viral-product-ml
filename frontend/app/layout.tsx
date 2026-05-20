import "./globals.css";
import Link from "next/link";
import { ReactNode } from "react";
import { getDataReferenceDate } from "@/lib/data";

export const metadata = {
  title: "VIRAL — 식품 키워드 바이럴 예측",
  description: "과자·베이커리 키워드의 바이럴 강도와 피크 시점 예측 대시보드",
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  const refDate = await getDataReferenceDate();
  return (
    <html lang="ko">
      <body className="min-h-screen">
        <header className="sticky top-0 z-30 backdrop-blur-md bg-bg/70 border-b border-border">
          <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center gap-6">
            <Link href="/" className="font-bold text-lg tracking-tight flex items-center gap-2">
              <span className="text-transparent bg-clip-text bg-gradient-to-br from-accent to-accent2">VIRAL</span>
              <span className="text-sub text-xs font-normal">food keyword forecaster</span>
            </Link>
            <nav className="flex gap-1 text-sm text-sub">
              <Link href="/" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">랭킹</Link>
              <Link href="/forecast" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">미래궤적</Link>
              <Link href="/peak-timing" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">피크시점</Link>
              <Link href="/risk" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">하락주의</Link>
              <Link href="/explore" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">탐색</Link>
              <Link href="/compare" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">비교</Link>
              <Link href="/category/베이커리" className="px-3 py-1.5 rounded-lg hover:bg-panel2 hover:text-text">카테고리</Link>
            </nav>
            <div className="ml-auto flex items-center gap-3 text-xs">
              {refDate && (
                <span className="px-2.5 py-1 rounded-md border border-border bg-panel2/50 text-sub">
                  데이터 기준 <span className="text-text font-mono">{refDate}</span>
                </span>
              )}
            </div>
          </div>
        </header>
        <main className="max-w-[1400px] mx-auto px-6 py-8">{children}</main>
        <footer className="border-t border-border mt-16 py-8 text-center text-xs text-sub">
          캡스톤 — LightGBM + LSTM/seq2seq 앙상블 바이럴 예측 시스템
        </footer>
      </body>
    </html>
  );
}
