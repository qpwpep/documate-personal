import { expect, test, type Locator } from '@playwright/test';

async function expectClickableInViewport(locator: Locator) {
  await expect(locator).toBeVisible();
  await expect.poll(() => locator.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const atCenter = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
    return rect.top >= 0 && rect.left >= 0 && rect.bottom <= innerHeight
      && rect.right <= innerWidth && atCenter !== null && element.contains(atCenter);
  })).toBe(true);
}

test('shrinking the desktop sidebar returns keyboard focus to its visible menu opener', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: '새 채팅', exact: true }).focus();
  await page.setViewportSize({ width: 390, height: 844 });
  const opener = page.getByRole('button', { name: '메뉴 열기', exact: true });
  await expect(opener).toBeFocused();
  await expectClickableInViewport(opener);
});

test('evidence opens with the first pointer click after changing viewport width', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('./');
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'resize-source.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 1\r\n'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  const query = page.getByRole('textbox', { name: '질문', exact: true });
  await query.fill('화면 크기 변경 후 첨부 원문 확인');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  const source = page.getByRole('button', { name: '근거 [1] resize-source.py', exact: true });
  await expect(source).toBeVisible();
  const dialog = page.getByRole('dialog', { name: 'resize-source.py', exact: true });
  for (const viewport of [
    { width: 390, height: 844 }, { width: 1280, height: 720 },
    { width: 390, height: 360 }, { width: 1280, height: 720 },
  ]) {
    await query.focus();
    await page.setViewportSize(viewport);
    // Keep pointer down across a frame, as a normal physical click can do.
    await source.click({ delay: 60 });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole('group', { name: '원문 1행', exact: true })).toContainText('alpha = 1');
    await dialog.getByRole('button', { name: '근거 닫기', exact: true }).click();
    await expect(dialog).toHaveCount(0);
  }
});

for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 600 }]) {
  test(`answer keyboard focus stays below the sticky header at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto('./');
    const query = page.getByRole('textbox', { name: '질문', exact: true });
    for (let index = 0; index < 5; index += 1) {
      await query.fill(`키보드 탐색 확인 ${index}`);
      await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
      await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(index + 1);
    }
    const controls = (await page.getByRole('region', { name: '대화 기록', exact: true }).all()).flatMap((turn) => [
      turn.getByRole('button', { name: 'TXT 파일 만들기', exact: true }),
      turn.getByText('답변 정보', { exact: true }),
    ]);
    expect(controls).toHaveLength(10);
    await query.fill('아직 보내지 않은 후속 질문');
    const jump = page.getByRole('button', { name: '최신 답변으로 이동', exact: true });
    if (await jump.isVisible()) {
      await page.keyboard.press('Shift+Tab');
      await expect(jump).toBeFocused();
      await expectClickableInViewport(jump);
    }
    for (const { key, targets } of [
      { key: 'Shift+Tab', targets: [...controls].reverse() },
      { key: 'Tab', targets: controls.slice(1) },
    ]) {
      for (const target of targets) {
        await page.keyboard.press(key);
        await expect(target).toBeFocused();
        await expectClickableInViewport(target);
        await expect.poll(async () => {
          const header = await page.locator('.app-header').boundingBox();
          const focused = await target.boundingBox();
          return focused!.y >= header!.y + header!.height;
        }, { message: `${key}: ${await target.textContent()} must not be hidden behind the header` }).toBe(true);
      }
    }
    if (await jump.isVisible()) {
      await page.keyboard.press('Tab');
      await expect(jump).toBeFocused();
      await expectClickableInViewport(jump);
    }
    await page.keyboard.press('Tab');
    await expect(query).toBeFocused();
    await expect(query).toHaveValue('아직 보내지 않은 후속 질문');
  });
}

for (const viewport of [{ width: 320, height: 720 }, { width: 390, height: 844 }, { width: 390, height: 360 }]) {
  test(`keyboard navigation and evidence remain usable at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto('./');
    const menu = page.getByRole('button', { name: '메뉴 열기', exact: true });
    await menu.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('button', { name: '메뉴 닫기', exact: true })).toBeVisible();
    await page.keyboard.press('Shift+Tab');
    await expect.poll(() => page.getByRole('dialog', { name: '대화 메뉴', exact: true })
      .evaluate((element) => element.contains(document.activeElement))).toBe(true);
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: '메뉴 닫기', exact: true })).toBeFocused();
    await page.getByText('설정', { exact: true }).click();
    await page.getByText('화면 설정', { exact: true }).click();
    for (const theme of ['라이트', '다크', '시스템']) {
      await page.getByRole('radio', { name: theme, exact: true }).check();
      await expect(page.getByRole('radio', { name: theme, exact: true })).toBeChecked();
    }
    await page.keyboard.press('Escape');
    await expect(menu).toBeFocused();

    await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
      name: 'focus.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 1\r\n'),
    });
    await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
    await page.getByRole('textbox', { name: '질문', exact: true }).fill('화면과 근거 키보드 검증');
    const send = page.getByRole('button', { name: '질문 보내기', exact: true });
    await expect(send).toBeEnabled();
    await send.click();
    const source = page.getByRole('button', { name: '근거 [1] focus.py', exact: true });
    await expect(source).toBeVisible();
    await source.focus();
    await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog', { name: 'focus.py', exact: true });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole('group', { name: '원문 1행', exact: true })).toContainText('alpha = 1');
    await expectClickableInViewport(dialog.getByRole('button', { name: '근거 닫기', exact: true }));
    await page.keyboard.press('Tab');
    await expect.poll(() => dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
    await page.keyboard.press('Shift+Tab');
    await expect.poll(() => dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
    await page.setViewportSize({ width: 1280, height: 720 });
    const sourceDetails = dialog.getByText('출처와 수집 정보', { exact: true });
    await sourceDetails.focus();
    await page.setViewportSize(viewport);
    await expect(sourceDetails).toBeFocused();
    await expectClickableInViewport(sourceDetails);
    await page.screenshot({ path: test.info().outputPath('evidence-open.png'), fullPage: true });
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
    await expect(source).toBeFocused();
    await expectClickableInViewport(source);
    await source.click();
    await dialog.getByRole('button', { name: '근거 닫기', exact: true }).click();
    await expect(source).toBeFocused();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: test.info().outputPath('mobile-evidence.png'), fullPage: true });
  });
}

test('light, dark and system themes keep text, code, warnings and action labels readable', async ({ page }) => {
  await page.goto('./');
  await page.getByText('설정', { exact: true }).click();
  await page.getByText('화면 설정', { exact: true }).click();
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('테마 대비 확인');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'contrast.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 1\r\n'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('코드 대비 확인');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  const source = page.getByRole('button', { name: '근거 [1] contrast.py', exact: true });
  await expect(source).toBeVisible();
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('제공자 오류 검증: 경고 대비 확인');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.locator('.result-failed')).toBeVisible();
  await page.route('**/agent/stream', (route) => route.abort('failed'));
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('연결 경고 대비 확인');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.locator('.turn .notice')).toBeVisible();
  for (const theme of ['라이트', '다크', '시스템']) {
    if (theme === '시스템') await page.emulateMedia({ colorScheme: 'dark' });
    await page.getByRole('radio', { name: theme, exact: true }).check();
    await source.click();
    await expect(page.locator('.evidence-code-line code')).toBeVisible();
    const ratios = await page.locator('.user-message p, .message-label, .result-status, .result-message, .notice, .answer-body p, .answer-body code, .evidence-code-line code, .evidence-code-line mark, .evidence-caption, .new-conversation, .secondary-button').evaluateAll((elements) => {
      const rgb = (value: string) => value.match(/[\d.]+/g)!.map(Number);
      const luminance = (channels: number[]) => channels.slice(0, 3).map((channel) => {
        const normalized = channel / 255;
        return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
      }).reduce((sum, channel, index) => sum + channel * [0.2126, 0.7152, 0.0722][index], 0);
      return elements.filter((element) => element.getClientRects().length > 0).map((element) => {
        let ancestor: Element | null = element;
        let background = [255, 255, 255];
        while (ancestor) {
          const color = rgb(getComputedStyle(ancestor).backgroundColor);
          if (color.length === 3 || color[3] === 1) { background = color; break; }
          ancestor = ancestor.parentElement;
        }
        const foreground = luminance(rgb(getComputedStyle(element).color));
        const behind = luminance(background);
        return { text: element.textContent, ratio: (Math.max(foreground, behind) + 0.05) / (Math.min(foreground, behind) + 0.05) };
      });
    });
    expect(ratios.length).toBeGreaterThan(4);
    for (const { text, ratio } of ratios) expect(ratio, `${theme}: ${text}`).toBeGreaterThanOrEqual(4.5);
    await page.getByRole('button', { name: '근거 닫기', exact: true }).click();
  }
});

test('a focused excerpt search remains visible when the open evidence panel becomes short', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('./');
  const notebook = { nbformat: 4, nbformat_minor: 5, metadata: {}, cells: [{
    cell_type: 'markdown', id: 'long-source', metadata: {},
    source: Array.from({ length: 32 }, (_, index) => `alpha 자료 ${index + 1}: 원문에서 필요한 위치를 찾고 확인합니다.\n\n`),
  }] };
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'long-source.ipynb', mimeType: 'application/x-ipynb+json', buffer: Buffer.from(JSON.stringify(notebook)),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('긴 발췌 검색 키보드 검증');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await page.getByRole('button', { name: '근거 [1] long-source.ipynb', exact: true }).click();
  const search = page.getByRole('searchbox', { name: '발췌에서 찾기', exact: true });
  await search.fill('자료 3');
  await page.setViewportSize({ width: 390, height: 360 });
  await expect(search).toBeFocused();
  await expectClickableInViewport(search);
  await expect(page.getByText(/검색어와 일치하는 문단/)).toBeVisible();
});

test('receiving another answer preserves the position of an earlier answer being read', async ({ page, request }) => {
  await page.goto('./');
  for (let index = 0; index < 4; index += 1) {
    await page.getByRole('textbox', { name: '질문', exact: true }).fill(`읽기 위치 이전 질문 ${index}`);
    await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
    await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(index + 1);
  }
  const query = '읽는 동안 뒤에서 받는 응답';
  await request.post('http://127.0.0.1:18000/_fixture/hold', { data: { query } });
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByRole('button', { name: '응답 수신 중단', exact: true })).toBeVisible();
  const earlier = page.getByText('읽기 위치 이전 질문 1', { exact: true });
  await earlier.scrollIntoViewIfNeeded();
  const before = await earlier.boundingBox();
  await request.post('http://127.0.0.1:18000/_fixture/release', { data: { query } });
  await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(5);
  expect(Math.abs((await earlier.boundingBox())!.y - before!.y)).toBeLessThanOrEqual(1);
});
