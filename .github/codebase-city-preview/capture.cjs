const { mkdir, rename, stat } = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");

const defaultCityUrl =
  "https://gam-codebase-city.sauerslabs.chatgpt.site/?capture=readme";
const output = path.resolve(
  process.argv[2] ?? "docs/images/codebase-city-latest.png",
);
const cityUrl = process.argv[3] ?? defaultCityUrl;
const temporary = `${output}.new`;

async function capture() {
  await mkdir(path.dirname(output), { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: [
      "--enable-webgl",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
      "--use-angle=swiftshader",
      "--use-gl=angle",
    ],
  });

  try {
    const page = await browser.newPage({
      viewport: { width: 1600, height: 900 },
      deviceScaleFactor: 1,
      colorScheme: "dark",
      locale: "en-US",
      timezoneId: "UTC",
    });
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));

    const response = await page.goto(cityUrl, {
      waitUntil: "domcontentloaded",
      timeout: 90_000,
    });
    if (!response?.ok()) {
      throw new Error(`City returned HTTP ${response?.status() ?? "unknown"}`);
    }

    await page.waitForSelector('[data-city-ready="true"] canvas', {
      state: "visible",
      timeout: 90_000,
    });
    await page.waitForFunction(
      () => {
        const canvas = document.querySelector('[data-city-ready="true"] canvas');
        const alert = document.querySelector('[role="alert"]');
        return (
          canvas instanceof HTMLCanvasElement &&
          canvas.width >= 1200 &&
          canvas.height >= 700 &&
          !alert
        );
      },
      { timeout: 30_000 },
    );
    await page.waitForTimeout(5_000);

    if (pageErrors.length > 0) {
      throw new Error(`City runtime error: ${pageErrors.join("; ")}`);
    }

    await page.screenshot({
      path: temporary,
      type: "png",
      fullPage: false,
      animations: "disabled",
      caret: "hide",
    });
    const image = await stat(temporary);
    if (image.size < 150_000) {
      throw new Error(
        `Captured image is suspiciously small (${image.size} bytes)`,
      );
    }
    await rename(temporary, output);
    process.stdout.write(`Captured ${image.size} bytes to ${output}\n`);
  } finally {
    await browser.close();
  }
}

capture().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
