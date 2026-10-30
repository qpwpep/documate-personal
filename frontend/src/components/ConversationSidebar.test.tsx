import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConversationSidebar } from './ConversationSidebar';

afterEach(() => vi.unstubAllGlobals());

function renderSidebar(mobile = false) {
  vi.stubGlobal('matchMedia', () => ({ matches: mobile, addEventListener() {}, removeEventListener() {} }));
  return render(<>
    <ConversationSidebar items={[{ id: 'one', title: '새 대화', busy: false, expired: false }]}
      activeId="one" onSelect={() => {}} onNew={() => {}} disabled={false} theme="system"
      onThemeChange={() => {}} onModalChange={() => {}}>
      <fieldset><legend>수신자 유형</legend>
        <label><input type="radio" name="recipient" value="omitted" defaultChecked />직접 지정 안 함</label>
        <label><input type="radio" name="recipient" value="channel" />채널</label>
        <label><input type="radio" name="recipient" value="user" />사용자</label>
        <label><input type="radio" name="recipient" value="email" />이메일</label>
      </fieldset>
      <button type="button" disabled>기본 수신자 다시 확인</button>
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
    const close = screen.getByRole('button', { name: '메뉴 닫기' });
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
    screen.getByRole('radio', { name: '직접 지정 안 함' }).focus();
    await user.tab();
    expect(screen.getByRole('button', { name: '본문 버튼' })).toHaveFocus();
  });
});
