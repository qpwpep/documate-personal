import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { ConversationState } from '../conversations';

interface Props {
  items: { id: string; title: string; busy: boolean }[];
  activeId: string;
  onSelect(id: string): void;
  onNew(): void;
  disabled: boolean;
  theme: ConversationState['theme'];
  onThemeChange(theme: ConversationState['theme']): void;
  onModalChange(open: boolean): void;
  children?: ReactNode;
}

export function ConversationSidebar({ items, activeId, onSelect, onNew, disabled, theme, onThemeChange, onModalChange, children }: Props) {
  const [mobile, setMobile] = useState(() => window.matchMedia?.('(max-width: 760px)').matches ?? false);
  const [open, setOpen] = useState(false);
  const panel = useRef<HTMLElement>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const returnFocusAfterResize = useRef(false);

  useEffect(() => {
    const media = window.matchMedia?.('(max-width: 760px)');
    if (!media) return;
    function resized() {
      if (media!.matches && panel.current?.contains(document.activeElement)) returnFocusAfterResize.current = true;
      setMobile(media!.matches);
      if (!media!.matches) setOpen(false);
    }
    media.addEventListener('change', resized);
    return () => media.removeEventListener('change', resized);
  }, []);

  useEffect(() => {
    onModalChange(mobile && open);
    if (mobile && open) closeButton.current?.focus();
    if (mobile && !open && returnFocusAfterResize.current) {
      opener.current?.focus();
      returnFocusAfterResize.current = false;
    }
    return () => onModalChange(false);
  }, [mobile, open, onModalChange]);

  function close() {
    setOpen(false);
    opener.current?.focus();
  }
  function select(id: string) {
    onSelect(id);
    if (mobile) close();
  }

  return <>
    {mobile && <button ref={opener} type="button" className="sidebar-opener secondary-button" aria-label="메뉴 열기"
      aria-expanded={open} aria-controls="conversation-menu" onClick={() => setOpen(true)}>☰</button>}
    {mobile && open && <button type="button" className="sidebar-backdrop" aria-label="메뉴 배경 닫기" tabIndex={-1} onClick={close} />}
    <aside id="conversation-menu" ref={panel} className={`conversation-sidebar${mobile ? ' mobile-sidebar' : ''}`}
      hidden={mobile && !open} aria-label="대화 메뉴" role={mobile ? 'dialog' : undefined} aria-modal={mobile && open ? true : undefined}
      onKeyDown={(event) => {
        if (!mobile || !open) return;
        if (event.key === 'Escape') { event.preventDefault(); close(); return; }
        if (event.key !== 'Tab') return;
        const controls = [...panel.current!.querySelectorAll<HTMLElement>('button:not(:disabled), select:not(:disabled), input:not(:disabled), textarea:not(:disabled), a[href], summary')]
          .filter((element) => !element.closest('[hidden]') && !insideClosedDetails(element));
        const first = controls[0];
        const last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }}>
      <div className="sidebar-heading"><strong>내 대화</strong>{mobile && <button ref={closeButton} type="button" className="secondary-button" onClick={close}>메뉴 닫기</button>}</div>
      <button type="button" className="new-conversation" disabled={disabled} onClick={() => { onNew(); if (mobile) close(); }}>새 채팅</button>
      <nav aria-label="최근 대화">
        {[...items].reverse().slice(0, 5).map((item) => <button type="button" className="conversation-choice" key={item.id}
          disabled={disabled} aria-current={item.id === activeId ? 'page' : undefined} aria-label={`대화 선택: ${item.title}`}
          onClick={() => select(item.id)}><span>{item.title}</span>{item.busy && <small>응답 받는 중</small>}</button>)}
      </nav>
      <details className="all-conversations"><summary>전체 대화 {items.length}개</summary>
        <label htmlFor="all-conversations">대화 선택</label>
        <select id="all-conversations" value={activeId} disabled={disabled} onChange={(event) => select(event.target.value)}>
          {[...items].reverse().map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
        </select>
      </details>
      <details className="theme-settings"><summary>화면 설정</summary><label htmlFor="theme">테마</label>
        <select id="theme" value={theme} onChange={(event) => onThemeChange(event.target.value as ConversationState['theme'])}>
          <option value="system">시스템</option><option value="light">라이트</option><option value="dark">다크</option>
        </select>
      </details>
      {children}
    </aside>
  </>;
}

function insideClosedDetails(element: HTMLElement): boolean {
  const details = element.closest('details:not([open])');
  return details !== null && details.querySelector('summary') !== element;
}
