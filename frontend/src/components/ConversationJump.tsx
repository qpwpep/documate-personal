import { useEffect, useState, type RefObject } from 'react';

export function ConversationJump({ conversation, composer, input }: {
  conversation: RefObject<HTMLElement | null>;
  composer: RefObject<HTMLElement | null>;
  input: RefObject<HTMLTextAreaElement | null>;
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const content = conversation.current;
    const footer = composer.current;
    if (!content || !footer) return;
    let frame = 0;
    function measure() {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const lastTurn = content!.lastElementChild;
        setVisible(lastTurn !== null && lastTurn.getBoundingClientRect().bottom > footer!.getBoundingClientRect().top);
      });
    }
    // Content and the composer can grow independently while the reader stays put.
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(content);
    observer?.observe(footer);
    window.addEventListener('scroll', measure, { passive: true });
    window.addEventListener('resize', measure);
    measure();
    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener('scroll', measure);
      window.removeEventListener('resize', measure);
    };
  }, [conversation, composer]);

  if (!visible) return null;
  return <button type="button" className="jump-to-latest" aria-label="최신 답변으로 이동" onClick={() => {
    window.scrollTo({ top: document.documentElement.scrollHeight });
    input.current?.focus({ preventScroll: true });
  }}>최신 답변 <span aria-hidden="true">↓</span></button>;
}
