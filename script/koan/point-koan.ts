#!/usr/bin/env bun
/**
 * Points a koan checkout (the current directory) at a gpui-koan version.
 *
 *     point-koan.ts --version 0.2.7
 *         every `gpui-koan*` requirement in koan's manifests becomes `=0.2.7`
 *     point-koan.ts --version 0.2.7 --staged PATH
 *         the same, and `[patch.crates-io]` takes each staged crate from
 *         PATH (the workspace publish-gpui-koan.ts stages), so koan builds
 *         against crates not yet published
 *
 * Only `version = "..."` strings on `gpui-koan*` dependencies are rewritten;
 * everything else in the manifests is left byte for byte.
 */
import { readdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: { version: { type: "string" }, staged: { type: "string" } },
});
const version = values.version;
if (!version) throw new Error("--version is required");

const manifests = ["Cargo.toml", ...readdirSync("crates").map((c) => join("crates", c, "Cargo.toml"))].filter(
  existsSync,
);
// `gpui-koan-x = "=0.2.6"`, `{ package = "gpui-koan-x", version = "=0.2.6" ... }`
const pin = /((?:^|\n)\s*[\w-]+\s*=\s*\{[^}\n]*package\s*=\s*"gpui-koan[\w-]*"[^}\n]*version\s*=\s*")[^"]*(")/g;
const bare = /((?:^|\n)\s*gpui-koan[\w-]*\s*=\s*")[^"]*(")/g;
const named = /((?:^|\n)\s*gpui-koan[\w-]*\s*=\s*\{[^}\n]*version\s*=\s*")[^"]*(")/g;
for (const path of manifests) {
  const before = readFileSync(path, "utf8");
  const after = before
    .replace(pin, `$1=${version}$2`)
    .replace(bare, `$1=${version}$2`)
    .replace(named, `$1=${version}$2`);
  if (after !== before) {
    writeFileSync(path, after);
    console.log(`pinned gpui-koan ${version} in ${path}`);
  }
}

if (values.staged) {
  const staged = resolve(values.staged);
  const crates = readdirSync(staged).filter((d) => existsSync(join(staged, d, "Cargo.toml")));
  const lines = crates.map((dir) => {
    const name = /^name\s*=\s*"([^"]+)"/m.exec(readFileSync(join(staged, dir, "Cargo.toml"), "utf8"))![1];
    return `${name} = { path = "${join(staged, dir)}" }`;
  });
  const root = readFileSync("Cargo.toml", "utf8");
  const header = "[patch.crates-io]";
  const patched = root.includes(header)
    ? root.replace(header, `${header}\n${lines.join("\n")}`)
    : `${root.trimEnd()}\n\n${header}\n${lines.join("\n")}\n`;
  writeFileSync("Cargo.toml", patched);
  console.log(`patched ${lines.length} staged crates from ${staged}`);
}
