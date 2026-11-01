import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConversationSidebar } from './ConversationSidebar';

afterEach(() => vi.unstubAllGlobals());

function renderSidebar(mobile = false, props: Partial<ComponentProps<typeof ConversationSidebar>> = {}) {
  vi.stubGlobal('matchMedia', () => ({ matches: mobile, addEventListener() {}, removeEventListener() {} }));
  return render(<>
    <ConversationSidebar items={[{ id: 'one', title: '새 대화', busy: false, expired: false }]}
      activeId="one" onSelect={() => {}} onNew={() => {}} disabled={false} theme="system"
      onThemeChange={() => {}} onModalChange={() => {}} {...props}>
      <details><summary>Slack 수신자 설정</summary><fieldset><legend>수신자 유형</legend>
        <label><input type="radio" name="recipient" value="omitted" defaultChecked />직접 지정 안 함</label>
        <label><input type="radio" name="recipient" value="channel" />채널</label>
        <label><input type="radio" name="recipient" value="user" />사용자</label>
        <label><input type="radio" name="recipient" value="email" />이메일</label>
      </fieldset>
      <button type="button" disabled>기본 수신자 다시 확인</button></details>
    </ConversationSidebar>
    <button type="button">본문 버튼</button>
  </>);
}

describe('대화 메뉴 열기와 키보드 탐색', () => {
  it('데스크톱에서도 메뉴를 접고 다시 열며 조작 버튼으로 포커스를 옮긴다', async () => {
    const user = userEvent.setup();
    renderSidebar();
    expect(screen.getByRole('complementary', { name: '대화 메뉴' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: '메뉴 닫기' }));
    expect(screen.queryByRole('complementary', { name: '대화 메뉴' })).not.toBeInTheDocument();
    const open = screen.getByRole('button', { name: '메뉴 열기' });
    expect(open).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(screen.getByRole('complementary', { name: '대화 메뉴' })).toBeVisible();
    expect(screen.getByRole('button', { name: '메뉴 닫기' })).toHaveFocus();
  });

  it('모바일에서는 마지막 라디오 그룹의 선택 항목과 메뉴 시작 사이에서 Tab을 순환한다', async () => {
    const user = userEvent.setup();
    renderSidebar(true);
    await user.click(screen.getByRole('button', { name: '메뉴 열기' }));
    await user.click(screen.getByText('설정', { selector: 'summary' }));
    await user.click(screen.getByText('Slack 수신자 설정', { selector: 'summary' }));
    const close = screen.getByRole('button', { name: '메뉴 닫기' });
    close.focus();
    expect(close).toHaveFocus();
    await user.tab({ shift: true });
    expect(screen.getByRole('radio', { name: '직접 지정 안 함' })).toHaveFocus();
    await user.tab();
    expect(close).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.getByRole('button', { name: '메뉴 열기' })).toHaveFocus();
    expect(screen.queryByRole('dialog', { name: '대화 메뉴' })).not.toBeInTheDocument();
  });

  it('데스크톱에서는 마지막 라디오 그룹에서 본문으로 Tab 이동을 허용한다', async () => {
    const user = userEvent.setup();
    renderSidebar();
    await user.click(screen.getByText('설정', { selector: 'summary' }));
    await user.click(screen.getByText('Slack 수신자 설정', { selector: 'summary' }));
    screen.getByRole('radio', { name: '직접 지정 안 함' }).focus();
    await user.tab();
    expect(screen.getByRole('button', { name: '본문 버튼' })).toHaveFocus();
  });

  it('모바일에서 접힌 설정 안의 상세 메뉴를 건너뛰고 보이는 마지막 항목으로 순환한다', async () => {
    const user = userEvent.setup();
    renderSidebar(true);
    await user.click(screen.getByRole('button', { name: '메뉴 열기' }));
    await user.tab({ shift: true });
    expect(screen.getByText('설정', { selector: 'summary' })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('button', { name: '메뉴 닫기' })).toHaveFocus();
  });
});

describe('사이드바 정보 우선순위', () => {
  it('대화 제목은 번호와 현재 상태 문구 없이 표시한다', () => {
    renderSidebar();
    const current = screen.getByRole('button', { name: '대화 선택: 새 대화' });
    expect(current).toHaveTextContent(/^새 대화$/);
    expect(current).toHaveAttribute('aria-current', 'page');
  });

  it('첨부와 설정을 접어 두고 필요한 기능을 펼쳐서 사용한다', async () => {
    const user = userEvent.setup();
    const onThemeChange = vi.fn();
    renderSidebar(false, {
      attachments: <button type="button">첨부 목록 새로고침</button>, onThemeChange,
    });
    expect(screen.getByRole('button', { name: '첨부 목록 새로고침' })).not.toBeVisible();
    expect(screen.getByRole('radio', { name: '라이트' })).not.toBeVisible();
    await user.click(screen.getByText('첨부 자료', { selector: 'summary' }));
    expect(screen.getByRole('button', { name: '첨부 목록 새로고침' })).toBeVisible();
    await user.click(screen.getByText('설정', { selector: 'summary' }));
    await user.click(screen.getByText('화면 설정', { selector: 'summary' }));
    await user.click(screen.getByRole('radio', { name: '라이트' }));
    expect(onThemeChange).toHaveBeenCalledExactlyOnceWith('light');
    await user.click(screen.getByText('Slack 수신자 설정', { selector: 'summary' }));
    expect(screen.getByRole('radio', { name: '직접 지정 안 함' })).toBeVisible();
  });

  it('전체 대화 선택은 펼친 뒤 사용할 수 있다', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderSidebar(false, { onSelect, items: [
      { id: 'one', title: '새 대화', busy: false, expired: false },
      { id: 'two', title: '이전 대화', busy: false, expired: false },
    ] });
    expect(screen.getByRole('combobox', { name: '대화 선택' })).not.toBeVisible();
    await user.click(screen.getByText('전체 대화', { selector: 'summary' }));
    await user.selectOptions(screen.getByRole('combobox', { name: '대화 선택' }), 'two');
    expect(onSelect).toHaveBeenCalledExactlyOnceWith('two');
  });
});

describe('대화 제목 검색', () => {
  const items = [
    { id: 'one', title: 'React 공식 문서', busy: false, expired: false },
    { id: 'two', title: 'Python 함수 복습', busy: false, expired: false },
    { id: 'three', title: 'React Hooks 문서', busy: true, expired: false },
    { id: 'four', title: '이전 React 문서', busy: false, expired: true },
    { id: 'five', title: '첨부 파일 요약', busy: false, expired: false },
  ];

  it('최근 목록 밖의 대화까지 대소문자와 앞뒤 공백을 무시하고 찾는다', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderSidebar(false, { items, activeId: 'five', onSelect });
    expect(screen.queryByRole('button', { name: '대화 선택: React 공식 문서' })).not.toBeInTheDocument();
    await user.type(screen.getByRole('searchbox', { name: '대화 검색' }), ' rEaCt ');
    expect(screen.getByRole('navigation', { name: '대화 검색 결과' })).toBeVisible();
    expect(screen.getByRole('status')).toHaveTextContent('검색 결과 3개');
    expect(screen.queryByRole('button', { name: '대화 선택: Python 함수 복습' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '대화 선택: React Hooks 문서' })).toHaveTextContent('답변 준비 중');
    expect(screen.getByRole('button', { name: '대화 선택: 이전 React 문서' })).toHaveTextContent('읽기 전용');
    await user.click(screen.getByRole('button', { name: '대화 선택: React 공식 문서' }));
    expect(onSelect).toHaveBeenCalledExactlyOnceWith('one');
  });

  it('결과가 없으면 안내하고 검색어를 지우면 최근 목록과 입력 포커스를 복원한다', async () => {
    const user = userEvent.setup();
    renderSidebar(false, { items, activeId: 'five' });
    const search = screen.getByRole('searchbox', { name: '대화 검색' });
    await user.type(search, '없는 제목');
    expect(screen.getByText('일치하는 대화가 없습니다.')).toBeVisible();
    await user.click(screen.getByRole('button', { name: '검색어 지우기' }));
    expect(search).toHaveValue('');
    expect(search).toHaveFocus();
    expect(screen.getByRole('navigation', { name: '최근 대화' })).toBeVisible();
    expect(screen.queryByText('일치하는 대화가 없습니다.')).not.toBeInTheDocument();
  });

  it('검색 중에도 현재 대화와 작업 중 탐색 제한을 유지한다', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderSidebar(false, { items, activeId: 'one', disabled: true, onSelect });
    await user.type(screen.getByRole('searchbox', { name: '대화 검색' }), 'React');
    const current = screen.getByRole('button', { name: '대화 선택: React 공식 문서' });
    expect(current).toHaveAttribute('aria-current', 'page');
    expect(current).toBeDisabled();
    const other = screen.getByRole('button', { name: '대화 선택: React Hooks 문서' });
    expect(other).toBeDisabled();
    await user.click(other);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('모바일 검색 결과를 선택하면 메뉴를 닫고 열기 버튼으로 돌아간다', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderSidebar(true, { items, activeId: 'five', onSelect });
    await user.click(screen.getByRole('button', { name: '메뉴 열기' }));
    await user.type(screen.getByRole('searchbox', { name: '대화 검색' }), 'React');
    await user.click(screen.getByRole('button', { name: '대화 선택: 이전 React 문서' }));
    expect(onSelect).toHaveBeenCalledExactlyOnceWith('four');
    expect(screen.queryByRole('dialog', { name: '대화 메뉴' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '메뉴 열기' })).toHaveFocus();
  });
});
