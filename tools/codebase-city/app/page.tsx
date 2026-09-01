import Script from "next/script";

export default function Home() {
  return (
    <>
      <div id="city-root">
        <main className="city-shell">
          <div className="loading-fallback" role="status">
            Surveying repository geography…
          </div>
        </main>
      </div>
      <Script
        id="city-runtime"
        type="module"
        src="/city-runtime/city.js"
        strategy="afterInteractive"
      />
    </>
  );
}
