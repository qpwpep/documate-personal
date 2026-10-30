import { readFile, writeFile } from 'node:fs/promises';
import openapiTS, { astToString } from 'openapi-typescript';

const input = new URL('../src/api/openapi.json', import.meta.url);
const output = new URL('../src/api/generated.ts', import.meta.url);
const schema = JSON.parse(await readFile(input, 'utf8'));
const events = schema.paths['/agent/stream'].post.responses['200'].content['text/event-stream']['x-sse-events'];
const eventProperties = [];
for (const [event, dataSchema] of Object.entries(events)) {
  const component = `AgentStreamEvent_${event}`;
  if (component in schema.components.schemas) throw new Error(`Generated event schema already exists: ${component}`);
  // Make the extension visible to openapi-typescript without changing the
  // exported server schema or maintaining another copy of event contracts.
  schema.components.schemas[component] = dataSchema;
  eventProperties.push(`    ${JSON.stringify(event)}: components["schemas"][${JSON.stringify(component)}];`);
}
const types = astToString(await openapiTS(schema, { defaultNonNullable: false }));
const eventMap = `\nexport interface AgentStreamEventData {\n${eventProperties.join('\n')}\n}\n`;
const source = '// Generated from openapi.json. Run npm run generate:api; do not edit.\n' + types + eventMap;

if (process.argv.includes('--check')) {
  const current = await readFile(output, 'utf8');
  if (current !== source) {
    process.stderr.write('API types are stale. Run npm run generate:api.\n');
    process.exitCode = 1;
  }
} else {
  await writeFile(output, source);
}
