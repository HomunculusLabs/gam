import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://gam-codebase-city.test/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the repository city and its social metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>GAM Codebase City<\/title>/i);
  assert.match(html, /Surveying repository geography/);
  assert.doesNotMatch(
    html,
    /\/assets\/City-[^"']+\.js/,
    "the Three.js city bundle must stay behind the client-only boundary",
  );
  assert.doesNotMatch(html, /https:\/\/null/);
  assert.match(
    html,
    /https:\/\/gam-codebase-city\.sauerslabs\.chatgpt\.site\/og\.png/,
  );
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project/i);
});

test("ships a populated city snapshot and bespoke preview image", async () => {
  const city = JSON.parse(
    await readFile(new URL("../public/city-data.json", import.meta.url), "utf8"),
  );

  assert.ok(city.files.length > 3_000);
  assert.ok(city.issues.length > 300);
  assert.ok(city.failures.length > 100);
  assert.ok(city.history.length >= 8);
  assert.ok(city.dependencies.length > 20);
  await access(new URL("../public/og.png", import.meta.url));
});

test("keeps graphics libraries out of the Cloudflare worker graph", async () => {
  const server = new URL("../dist/server/", import.meta.url);
  const files = await readdir(server, { recursive: true });
  const javascript = files.filter((file) => file.endsWith(".js"));
  const source = (
    await Promise.all(
      javascript.map((file) => readFile(new URL(file, server), "utf8")),
    )
  ).join("\n");

  assert.doesNotMatch(source, /LoadingManager|@react-three|three\/build/);
});
