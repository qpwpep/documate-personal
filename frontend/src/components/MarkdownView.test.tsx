import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MarkdownView } from './MarkdownView';

describe('MarkdownView', () => {
  it('renders headings, lists, tables and inline formatting without executing source HTML', () => {
    const { container } = render(<MarkdownView text={'## 제목\n\n첫 **강조** 문장.\n다음 줄.\n\n- `order` 순서\n\n| 이름 | 값 |\n| --- | --- |\n| x | 2 |\n\n<script>alert("source")</script>\n\n[unsafe](javascript:alert)'} />);

    expect(screen.getByRole('heading', { name: '제목', level: 2 })).toBeVisible();
    expect(screen.getByText('강조').tagName).toBe('STRONG');
    expect(screen.getByRole('listitem')).toHaveTextContent('order 순서');
    expect(screen.getByRole('cell', { name: '2' })).toBeVisible();
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(container.textContent).toContain('<script>alert("source")</script>');
  });

  it('resolves source links and images while preserving code', () => {
    const relative = '../reference/generated/numpy.concatenate.html#numpy.concatenate';
    const text = `[concatenate](${relative}) and [reference][api] and [section](#arrays).\n\n![diagram](../images/arrays.svg)\n\n\`[example](${relative})\`\n\n\`\`\`markdown\n[example](${relative})\n\`\`\`\n\n[api]: ${relative}`;
    const { container } = render(<MarkdownView text={text} sourceUri="https://numpy.org/doc/2.1/user/absolute_beginners.html" />);

    expect(screen.getByRole('link', { name: 'concatenate' })).toHaveAttribute('href', 'https://numpy.org/doc/2.1/reference/generated/numpy.concatenate.html#numpy.concatenate');
    expect(screen.getByRole('link', { name: 'reference' })).toHaveAttribute('href', 'https://numpy.org/doc/2.1/reference/generated/numpy.concatenate.html#numpy.concatenate');
    expect(screen.getByRole('link', { name: 'section' })).toHaveAttribute('href', 'https://numpy.org/doc/2.1/user/absolute_beginners.html#arrays');
    expect(screen.getByRole('img', { name: 'diagram' })).toHaveAttribute('src', 'https://numpy.org/doc/2.1/images/arrays.svg');
    expect(container.querySelector('pre code')?.textContent).toBe(`[example](${relative})\n`);
  });

  it('can render an inline unit inside a paragraph without adding a nested paragraph', () => {
    const { container, rerender } = render(<p>설명: <MarkdownView text="**강조**와 `__version__`" inline /></p>);
    expect(screen.getByText('강조').tagName).toBe('STRONG');
    expect(screen.getByText('__version__').tagName).toBe('CODE');
    expect(container.querySelectorAll('p')).toHaveLength(1);
    rerender(<p><MarkdownView text={'첫 줄\n둘째 줄\n\n다음 문단'} inline /></p>);
    expect(container.textContent).toContain('첫 줄\n둘째 줄');
    expect(container.querySelectorAll('br')).toHaveLength(2);
  });

  it('preserves inline boundary whitespace and whitespace-only units', () => {
    const { container, rerender } = render(<MarkdownView text="  원문  " inline />);
    expect(container.textContent).toBe('  원문  ');

    rerender(<MarkdownView text={'\n\t **강조**와 `  literal  ` \t\n\n'} inline />);
    expect(container.textContent).toBe('\n\t 강조와  literal  \t\n\n');
    expect(screen.getByText('강조').tagName).toBe('STRONG');
    expect(container.querySelector('code')?.textContent).toBe(' literal ');

    rerender(<MarkdownView text={' \t\n\n  '} inline />);
    expect(container.textContent).toBe(' \t\n\n  ');

    rerender(<MarkdownView text="" inline />);
    expect(container.textContent).toBe('');
  });

  it('separates only a provider heading permalink outside literal code and links', () => {
    const { container, rerender } = render(<MarkdownView text={'# `pandas.merge`#. pandas.merge(*left*, *right*)#. Join tables.\n\n## Examples\n\n```python\npd.merge(left, right)\n```'} providerExcerpt sourceUri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html" />);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('pandas.merge');
    expect(screen.getByRole('heading', { level: 1 })).not.toHaveTextContent('Join tables');
    expect(container.querySelector('pre code')?.textContent).toBe('pd.merge(left, right)\n');
    rerender(<MarkdownView text="# Read `value#. literal` and [value#. linked](https://example.org) safely" providerExcerpt sourceUri="https://example.org/guide" />);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Read value#. literal and value#. linked safely');
    rerender(<MarkdownView text="# Heading#. ordinary text" sourceUri="https://example.org/guide" />);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Heading#. ordinary text');
  });

  it('displays top-level source parameter definitions as prose and preserves adjacent code', () => {
    const text = "Parameters:\n:   **a**array\\_like\n    :   Array to reshape.\n\n    **shape**int or tuple\n    :   Target shape.\n\n    result = values.reshape((2, 3))\n\n```python\n    **literal**code\n    :   unchanged\n```";
    const { container } = render(<MarkdownView text={text} sourceUri="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html" />);
    expect(screen.getByText('a').closest('p')).toHaveTextContent('a array_like');
    expect(screen.getByText('shape', { selector: 'strong' }).closest('p')).toHaveTextContent('shape int or tuple');
    expect(screen.getByText('Target shape.').tagName).toBe('P');
    const code = [...container.querySelectorAll('pre code')].map((element) => element.textContent);
    expect(code).toContain('result = values.reshape((2, 3))\n');
    expect(code).toContain('    **literal**code\n    :   unchanged\n');
  });

  it('keeps parameter-looking content inside list and quote containers', () => {
    const { container } = render(<MarkdownView text={'- **shape**int\n  :   Target shape.\n\n- Keep item.\n\n> **order**str\n> :   Traversal.'} sourceUri="https://example.org/guide" />);
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
    expect(container.querySelector('blockquote')).toHaveTextContent('orderstr');
    expect(container.querySelector('blockquote')?.textContent).toContain(':   Traversal.');
  });

  it('reads retained parameter prose with another indentation level', () => {
    const text = 'numpy.reshape(*a*, *shape*)\n:   Returns a reshaped ndarray.\n\n    Parameters:\n    :   **a**array\\_like\n        :   Array to reshape.\n\n        **shape**int or tuple\n        :   Target shape.\n\n        **order**{C, F}, optional\n        :   Read elements of `a` using this index order.';
    render(<MarkdownView text={text} sourceUri="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html" />);
    expect(screen.getByText('a', { selector: 'strong' }).closest('p')).toHaveTextContent('a array_like');
    expect(screen.getByText('shape', { selector: 'strong' }).closest('p')).toHaveTextContent('shape int or tuple');
    expect(screen.getByText('order').closest('p')).toHaveTextContent('order {C, F}, optional');
    expect(screen.getByText('a', { selector: 'code' })).toBeVisible();
  });
});
