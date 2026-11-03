#!/usr/bin/env bun
/**
 * The gpui-koan version, kept in step across script/koan/VERSION, the
 * renamed crates' manifests and their Cargo.lock entries, so a git
 * dependency on this branch satisfies the same requirement a published
 * release does.
 *
 *     set-version.ts 0.2.1    moves every one of them to 0.2.1
 *     set-version.ts --check  fails where one differs from VERSION
 *
 * Run from the repository's root.
 */
import { readdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const PREFIX = "gpui-koan";
const arg = process.argv[2];
if (arg === undefined) throw new Error("usage: set-version.ts VERSION | --check");
const check = arg === "--check";
const version = check ? readFileSync("script/koan/VERSION", "utf8").trim() : arg;
if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error(`\`${version}\` is not a version`);

const wrong: string[] = [];
const renamed: string[] = [];
for (const dir of readdirSync("crates")) {
  const path = join("crates", dir, "Cargo.toml");
  if (!existsSync(path)) continue;
  const text = readFileSync(path, "utf8");
  const pkg = /^\[package\]\n([\s\S]*?)(?=^\[|(?![\s\S]))/m.exec(text);
  const name = pkg && /^name = "([^"]+)"$/m.exec(pkg[1])?.[1];
  if (!pkg || !name?.startsWith(PREFIX)) continue;
  renamed.push(name);
  const current = /^version = "([^"]+)"$/m.exec(pkg[1])?.[1];
  if (current === version) continue;
  if (check) {
    wrong.push(`${path}: ${current}`);
    continue;
  }
  const body = pkg[1].replace(/^version = "[^"]+"$/m, `version = "${version}"`);
  writeFileSync(path, text.slice(0, pkg.index) + "[package]\n" + body + text.slice(pkg.index + pkg[0].length));
}
if (renamed.length === 0) throw new Error(`no ${PREFIX} crates under crates/`);

const lock = readFileSync("Cargo.lock", "utf8");
let updated = lock;
for (const name of renamed) {
  const entry = new RegExp(`^(name = "${name}"\\nversion = )"([^"]+)"$`, "m");
  const found = entry.exec(updated);
  if (found === null) {
    wrong.push(`Cargo.lock: no entry for ${name}`);
    continue;
  }
  if (found[2] === version) continue;
  if (check) wrong.push(`Cargo.lock ${name}: ${found[2]}`);
  else updated = updated.replace(entry, `$1"${version}"`).replaceAll(` "${name} ${found[2]}"`, ` "${name} ${version}"`);
}

if (check) {
  if (wrong.length > 0) {
    console.error(`out of step with VERSION ${version}:\n  ${wrong.join("\n  ")}`);
    process.exit(1);
  }
  console.log(`${renamed.length} crates at ${version}`);
} else {
  if (wrong.length > 0) throw new Error(wrong.join("\n"));
  writeFileSync("Cargo.lock", updated);
  writeFileSync("script/koan/VERSION", `${version}\n`);
  console.log(`${renamed.length} crates moved to ${version}`);
}
