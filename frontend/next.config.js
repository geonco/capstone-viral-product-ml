/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  // 서버 컴포넌트가 public/mock/*.json을 fs.readFile로 읽음
  // Vercel output tracing이 동적 경로를 못 잡아서 누락되는 걸 방지
  experimental: {
    outputFileTracingIncludes: {
      "/**": ["./public/mock/**/*"],
    },
  },
};
