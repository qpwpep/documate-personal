import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { ConversationState } from '../conversations';
import './ConversationSidebar.css';

interface Props {
  items: { id: string; title: string; busy: boolean; expired: boolean }[];
  activeId: string;
  onSelect(id: string): void;
  onNew(): void;
  disabled: boolean;
  theme: ConversationState['theme'];
  onThemeChange(theme: ConversationState['theme']): void;
  onModalChange(open: boolean): void;
  attachments?: ReactNode;
  help?: ReactNode;
  children?: ReactNode;
}

export function ConversationSidebar({ items, activeId, onSelect, onNew, disabled, theme, onThemeChange, onModalChange, attachments, help, children }: Props) {
  const [mobile, setMobile] = useState(() => window.matchMedia?.('(max-width: 760px)').matches ?? false);
  const [open, setOpen] = useState(() => !mobile);
  const [search, setSearch] = useState('');
  const panel = useRef<HTMLElement>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const returnFocusToOpener = useRef(false);
  const focusOnOpen = useRef(false);
  const numberedItems = items.map((item, index) => ({ ...item,
    label: `${index + 1}. ${item.title}${item.busy ? ' · 답변 준비 중' : item.expired ? ' · 읽기 전용' : ''}`,
  }));
  const recentItems = numberedItems.slice(-3).reverse();
  const currentItem = numberedItems.find((item) => item.id === activeId);
  if (currentItem && !recentItems.some((item) => item.id === activeId)) recentItems.push(currentItem);
  const searchTerm = search.trim().toLocaleLowerCase();
  const visibleItems = searchTerm
    ? numberedItems.filter((item) => item.title.toLocaleLowerCase().includes(searchTerm)).reverse()
    : recentItems;

  useEffect(() => {
    const media = window.matchMedia?.('(max-width: 760px)');
    if (!media) return;
    function resized() {
      if (media!.matches && panel.current?.contains(document.activeElement)) returnFocusToOpener.current = true;
      setMobile(media!.matches);
      setOpen(!media!.matches);
    }
    media.addEventListener('change', resized);
    return () => media.removeEventListener('change', resized);
  }, []);

  useEffect(() => {
    onModalChange(mobile && open);
    if (open && focusOnOpen.current) {
      closeButton.current?.focus();
      focusOnOpen.current = false;
    }
    if (!open && returnFocusToOpener.current) {
      opener.current?.focus();
      returnFocusToOpener.current = false;
    }
    return () => onModalChange(false);
  }, [mobile, open, onModalChange]);

  function close() {
    returnFocusToOpener.current = true;
    setOpen(false);
  }
  function select(id: string) {
    onSelect(id);
    if (mobile) close();
  }

  return <>
    {!open && <button ref={opener} type="button" className="sidebar-opener secondary-button" aria-label="메뉴 열기"
      aria-expanded={open} aria-controls="conversation-menu" onClick={() => { focusOnOpen.current = true; setOpen(true); }}>☰</button>}
    {mobile && open && <button type="button" className="sidebar-backdrop" aria-label="메뉴 배경 닫기" tabIndex={-1} onClick={close} />}
    <aside id="conversation-menu" ref={panel} className={`conversation-sidebar${mobile ? ' mobile-sidebar' : ''}`}
      hidden={!open} aria-label="대화 메뉴" role={mobile ? 'dialog' : undefined} aria-modal={mobile && open ? true : undefined}
      onKeyDown={(event) => {
        if (!mobile || !open) return;
        if (event.key === 'Escape') { event.preventDefault(); close(); return; }
        if (event.key !== 'Tab') return;
        const controls = [...panel.current!.querySelectorAll<HTMLElement>('button:not(:disabled), select:not(:disabled), input:not(:disabled), textarea:not(:disabled), a[href], summary')]
          .filter((element) => !element.closest('[hidden]') && !insideClosedDetails(element)
            && (!(element instanceof HTMLInputElement) || element.type !== 'radio' || element.checked));
        const first = controls[0];
        const last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }}>
      <div className="sidebar-heading"><div className="brand"><span className="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="M12 6.5C9 4.5 5.5 4 3 5v14c3-1 6-.5 9 1.5 3-2 6-2.5 9-1.5V5c-2.5-1-6-.5-9 1.5Z" /><path d="M12 6.5v14" /></svg>
      </span><strong>DocuMate</strong></div><button ref={closeButton} type="button" className="sidebar-close secondary-button" aria-label="메뉴 닫기" onClick={close}>
        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="3" y="4" width="18" height="16" rx="3" /><path d="M9 4v16" /></svg>
      </button></div>
      <button type="button" className="new-conversation" disabled={disabled} onClick={() => { onNew(); if (mobile) close(); }}>
        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="m15 4 5 5M4 20l5-1 12-12a2 2 0 0 0-5-5L4 14v6Z" /></svg>
        새 채팅
      </button>
      <div className="conversation-search">
        <label className="visually-hidden" htmlFor="conversation-search">대화 검색</label>
        <input ref={searchInput} id="conversation-search" type="search" value={search} placeholder="대화 검색"
          autoComplete="off" aria-controls="conversation-list" onChange={(event) => setSearch(event.currentTarget.value)} />
        {search && <button type="button" className="search-clear" aria-label="검색어 지우기"
          onClick={() => { setSearch(''); searchInput.current?.focus(); }}>지우기</button>}
      </div>
      <nav id="conversation-list" aria-label={searchTerm ? '대화 검색 결과' : '최근 대화'}>
        <p className="sidebar-section-label" role={searchTerm ? 'status' : undefined}>{searchTerm ? `검색 결과 ${visibleItems.length}개` : '최근 대화'}</p>
        {searchTerm && visibleItems.length === 0 && <div className="conversation-search-empty">
          <p>일치하는 대화가 없습니다.</p><p>다른 제목으로 검색하거나 새 채팅을 시작하세요.</p>
        </div>}
        {visibleItems.map((item) => <button type="button" className="conversation-choice" key={item.id}
          disabled={disabled || item.id === activeId} aria-current={item.id === activeId ? 'page' : undefined} aria-label={`대화 선택: ${item.title}`}
          onClick={() => select(item.id)}><span>{item.title}</span>
          {(item.busy || item.expired) && <small>{item.busy ? '답변 준비 중' : '읽기 전용'}</small>}
        </button>)}
      </nav>
      <details className="all-conversations"><summary>전체 대화</summary>
        <label htmlFor="all-conversations">대화 선택</label>
        <select id="all-conversations" value={activeId} disabled={disabled} onChange={(event) => select(event.target.value)}>
          {[...numberedItems].reverse().map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
        <p className="sidebar-help">이 탭에 대화와 초안이 보관됩니다. 서버 세션이 만료되면 기록만 읽을 수 있습니다.</p>
      </details>
      <div className="sidebar-utilities">
        {attachments && <details className="sidebar-attachments"><summary>첨부 자료</summary>{attachments}</details>}
        <details className="sidebar-settings"><summary>설정</summary>
          <details className="theme-settings"><summary>화면 설정</summary>
            <fieldset className="settings-options" disabled={disabled}><legend>테마</legend>
              {(['system', 'light', 'dark'] as const).map((value, index) => <label key={value}>
                <input type="radio" name="theme" value={value} checked={theme === value} onChange={() => onThemeChange(value)} />
                {['시스템', '라이트', '다크'][index]}
              </label>)}
            </fieldset>
          </details>
          {children}
        </details>
        {help}
      </div>
    </aside>
  </>;
}

function insideClosedDetails(element: HTMLElement): boolean {
  let details = element.closest('details:not([open])');
  while (details) {
    if (details.querySelector(':scope > summary') !== element) return true;
    details = details.parentElement?.closest('details:not([open])') ?? null;
  }
  return false;
}
