/* wave-utils.js — SHARED helpers for all web/apps demos (vanilla JS, no build).
 * Mirrors the Julia side (specs/Structure.md §5). Top-level must stay DOM-free
 * so the file is node-parseable (`node -e "new Function(...)"` gate). All DOM
 * access happens inside functions only.
 *
 * MIRROR DISCLAIMER: values are duplicated from src/Core/Constants.jl with the
 * `// MIRROR ... — TASK-02.3, change both` header once TASK-02.3 lands. Until
 * then these are PLACEHOLDERS (see specs/Tasks.md TASK-00.5 Result).
 */

"use strict";

const SOV_DEFAULTS = {
  // TODO(02.3): replace with exact mirror of src/Core/Constants.jl values.
  hbar: 1.0,
  mass: 1.0,
  g: 1.0,
  V0: 1.0,
  a: 1.0,
  k0: 2 * Math.PI,
  seed: 1,
};

// --- assert helper: logs to console + paints an on-page badge -----------------
// Usage: sovAssert(cond, "message"); ends with sovAssertSummary() to render the
// final PASS/FAIL badge. The badge element is created lazily (DOM-free until
// called), which keeps this module node-parseable.
const _sovResults = { pass: 0, fail: 0, msgs: [] };

function sovAssert(cond, msg) {
  if (cond) {
    _sovResults.pass += 1;
    console.log("PASS:", msg);
  } else {
    _sovResults.fail += 1;
    _sovResults.msgs.push(msg);
    console.error("FAIL:", msg);
  }
  return cond;
}

function sovAssertSummary(containerId) {
  const total = _sovResults.pass + _sovResults.fail;
  const ok = _sovResults.fail === 0;
  const el =
    document.getElementById(containerId) || document.createElement("div");
  if (!el.id) {
    el.id = containerId || "sov-assert-badge";
    document.body.appendChild(el);
  }
  el.className = "sov-assert-badge " + (ok ? "sov-pass" : "sov-fail");
  el.textContent = ok
    ? "ASSERTS PASS " + _sovResults.pass + "/" + total
    : "ASSERTS FAIL " + _sovResults.fail + "/" + total +
      (_sovResults.msgs.length ? " — " + _sovResults.msgs[0] : "");
  return ok;
}

function sovResetAsserts() {
  _sovResults.pass = 0;
  _sovResults.fail = 0;
  _sovResults.msgs.length = 0;
}

// --- small numeric helpers -----------------------------------------------------
function sovFmt(x, n) {
  return Number(x).toFixed(n === undefined ? 4 : n);
}

// SNR in dB between a reference signal and a reconstruction
// (mirrors the SNR metric used in the Julia tournament — same formula).
function sovSnrDb(reference, reconstruction) {
  if (reference.length !== reconstruction.length) {
    throw new Error("sovSnrDb: length mismatch");
  }
  let sig = 0.0;
  let noise = 0.0;
  for (let i = 0; i < reference.length; i++) {
    sig += reference[i] * reference[i];
    const d = reference[i] - reconstruction[i];
    noise += d * d;
  }
  if (noise === 0) return Infinity;
  return 10 * Math.log10(sig / noise);
}

if (typeof module !== "undefined" && module.exports) {
  // Node-scaffold safety (never bundled; kept for syntax/smoke checks only).
  module.exports = { SOV_DEFAULTS, sovAssert, sovAssertSummary, sovFmt, sovSnrDb };
}