import type { Root, RootContent, PhrasingContent } from 'mdast';
import type { Processor } from 'unified';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownViewProps {
  text: string;
  sourceUri?: string;
  inline?: boolean;
  providerExcerpt?: boolean;
}

export function MarkdownView({ text, sourceUri, inline = false, providerExcerpt = false }: MarkdownViewProps) {
  // Structured inline units own their surrounding whitespace; Markdown treats
  // those same characters as block indentation or blank lines and removes them.
  const leadingWhitespace = inline ? text.match(/^\s*/)?.[0] ?? '' : '';
  const trailingWhitespace = inline ? text.slice(leadingWhitespace.length).match(/\s*$/)?.[0] ?? '' : '';
  const markdown = text.slice(leadingWhitespace.length, text.length - trailingWhitespace.length);
  const content = (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, [sourceReader, { source: Boolean(sourceUri), providerExcerpt }], [inlineParagraphs, { inline }]]}
      allowedElements={inline ? ['a', 'em', 'strong', 'code', 'del', 'br', 'img'] : undefined}
      unwrapDisallowed={inline}
      urlTransform={(url, key) => {
        let target = url;
        if (sourceUri) {
          try {
            target = new URL(url, sourceUri).href;
          } catch (error) {
            if (error instanceof TypeError) return '';
            throw error;
          }
        }
        if (/^[a-z][a-z0-9+.-]*:/i.test(target) && !/^(https?|mailto):/i.test(target)) return '';
        if (key === 'src' && /^mailto:/i.test(target)) return '';
        return defaultUrlTransform(target);
      }}
      components={{
        a: ({ href, title, children }) => href
          ? <a href={href} title={title} target="_blank" rel="noreferrer">{children}</a>
          : <span>{children}</span>,
        img: ({ src, alt, title }) => src
          ? <img src={src} alt={alt ?? ''} title={title} loading="lazy" style={{ maxWidth: '100%', height: 'auto' }} />
          : <span>{alt}</span>,
      }}
    >
      {markdown}
    </ReactMarkdown>
  );
  return inline
    ? <span className="dm-markdown-inline" style={{ whiteSpace: 'pre-wrap' }}>{leadingWhitespace}{content}{trailingWhitespace}</span>
    : <div className="dm-markdown" style={{ overflowWrap: 'anywhere' }}>{content}</div>;
}

function sourceReader(this: Processor, { source, providerExcerpt }: { source: boolean; providerExcerpt: boolean }) {
  return (tree: Root, file: { value: unknown }) => {
    if (!source) return;
    const captured = String(file.value);
    const displayNodes: RootContent[] = [];
    for (const node of tree.children) {
      // Only root nodes are adjusted: list/quote containers and fenced code
      // retain their Markdown meaning even when they resemble definitions.
      const raw = node.position?.start.offset !== undefined && node.position.end.offset !== undefined
        ? captured.slice(node.position.start.offset, node.position.end.offset)
        : '';
      if (node.type === 'paragraph' && hasDefinition(raw)) {
        displayNodes.push(...(this.parse(readableDefinitions(raw)) as Root).children);
      } else if (node.type === 'code' && !/^[ \t]*(?:`{3,}|~{3,})/.test(raw)) {
        const paragraphs = node.value.split(/\n{2,}/);
        if (paragraphs.some((paragraph) => hasDefinition(dedent(paragraph)))) {
          for (const paragraph of paragraphs) {
            const definition = dedent(paragraph);
            displayNodes.push(...(hasDefinition(definition)
              ? (this.parse(readableDefinitions(definition)) as Root).children
              : [{ ...node, value: paragraph }]));
          }
        } else {
          displayNodes.push(node);
        }
      } else {
        displayNodes.push(node);
      }
    }
    tree.children = providerExcerpt ? separateProviderHeadings(displayNodes) : displayNodes;
  };
}

function inlineParagraphs({ inline }: { inline: boolean }) {
  return (tree: Root) => {
    if (!inline) return;
    for (let index = 0; index < tree.children.length - 1; index += 1) {
      const node = tree.children[index];
      if (node.type === 'paragraph') node.children.push({ type: 'break' }, { type: 'break' });
    }
  };
}

function hasDefinition(text: string): boolean {
  return /^(?:[ \t]*:[ \t]+)?\*\*[^*\n]+\*\*[^\n]*\n[ \t]*:[ \t]+/m.test(text);
}

function readableDefinitions(text: string): string {
  return text.replace(/^[ \t]*:[ \t]+/gm, '\n').replace(/^(\*\*[^*\n]+\*\*)(?=[^\s:])/gm, '$1 ');
}

function dedent(text: string): string {
  const lines = text.split('\n');
  const margins = lines.filter((line) => line.trim()).map((line) => line.match(/^[ \t]*/)?.[0].length ?? 0);
  const margin = Math.min(...margins);
  return lines.map((line) => line.slice(margin)).join('\n');
}

function separateProviderHeadings(nodes: RootContent[]): RootContent[] {
  return nodes.flatMap((node): RootContent[] => {
    if (node.type !== 'heading') return [node];
    for (let index = 0; index < node.children.length; index += 1) {
      const child = node.children[index];
      if (child.type !== 'text') continue;
      const boundary = child.value.indexOf('#. ');
      if (boundary === -1) continue;
      const title: PhrasingContent[] = [...node.children.slice(0, index), { type: 'text', value: child.value.slice(0, boundary) }];
      const description: PhrasingContent[] = [{ type: 'text', value: child.value.slice(boundary + 3) }, ...node.children.slice(index + 1)];
      if (!hasContent(title) || !hasContent(description)) continue;
      return [{ ...node, children: title }, { type: 'paragraph', children: description }];
    }
    return [node];
  });
}

function hasContent(nodes: PhrasingContent[]): boolean {
  return nodes.some((node) => ('value' in node && Boolean(node.value.trim())) || ('children' in node && hasContent(node.children)));
}
