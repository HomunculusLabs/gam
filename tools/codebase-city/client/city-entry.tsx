import { createRoot } from "react-dom/client";
import { CodebaseCity } from "../app/City";

function mountCity() {
  const host = document.getElementById("city-root");
  if (!host) throw new Error("City mount point is missing");
  createRoot(host).render(<CodebaseCity />);
}

if (document.readyState === "complete") {
  mountCity();
} else {
  window.addEventListener("load", mountCity, { once: true });
}
