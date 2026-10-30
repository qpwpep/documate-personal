import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import semver from 'semver';

test('declared Node support satisfies every locked dependency', async () => {
  const manifest = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const lock = JSON.parse(await readFile(new URL('../package-lock.json', import.meta.url), 'utf8'));
  assert.equal(lock.packages[''].engines.node, manifest.engines.node);
  const incompatible = Object.entries(lock.packages)
    .filter(([path, dependency]) => path && dependency.engines?.node
      && !semver.subset(manifest.engines.node, dependency.engines.node))
    .map(([path, dependency]) => `${path}: ${dependency.engines.node}`);
  assert.deepEqual(incompatible, [], 'The declared Node range must be supported by all locked dependencies.');
});
