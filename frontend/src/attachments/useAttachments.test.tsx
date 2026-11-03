import { createServer } from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import * as buffer from 'node:buffer';
import { useState } from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { AgentSessionClient } from '../api/client';
import type { ClientConfig, UploadManifest } from '../api/types';
import { createAttachmentState } from './state';
import { useAttachments } from './useAttachments';

const hash = (text: string) => `sha256:${createHash('sha256').update(text).digest('hex')}`;
const config: ClientConfig = {
  upload: { accepted_suffixes: ['.py', '.ipynb'], max_files: 3, max_file_bytes: 100, max_total_bytes: 200 },
  query: { max_chars: 8192, max_utf8_bytes: 16384 },
};
const original = { file_id: 'original', name: 'sample.py', size_bytes: 3, content_hash: hash('old'), source_uri: 'upload:///attachments/original' };
let manifest: UploadManifest;
let stagedCount = 0;
let syncBodies: Record<string, unknown>[] = [];
let stagedFiles = new Map<string, { name: string; size_bytes: number; content_hash: string }>();
let failSync: 'none' | 'once' | 'conflict' = 'none';
let syncResult: UploadManifest | null = null;
let sentQueries: string[] = [];
let restoredQueries: string[] = [];
let apiBaseUrl = '';
let manifestGate: Promise<void> | null = null;

const server = createServer(async (request, response) => {
  response.setHeader('Content-Type', 'application/json');
  if (request.method === 'GET' && request.url?.endsWith('/uploads')) {
    await manifestGate;
    response.end(JSON.stringify(manifest));
    return;
  }
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  const bytes = Buffer.concat(chunks);
  if (request.url?.endsWith('/uploads/staging')) {
    // The server fake replaces HTTP only; actual FormData bytes are parsed here.
    const multipart = new Response(bytes, { headers: { 'Content-Type': request.headers['content-type'] ?? '' } });
    const form = await multipart.formData();
    const file = form.get('file');
    if (!(file instanceof File)) throw new Error('Expected a multipart file');
    const content = await file.text();
    const uploadId = String(++stagedCount).padStart(32, '0');
    const staged = { name: file.name, size_bytes: file.size, content_hash: hash(content) };
    stagedFiles.set(uploadId, staged);
    response.end(JSON.stringify({ upload_id: uploadId, ...staged }));
    return;
  }
  if (request.url?.endsWith('/uploads/sync')) {
    const body = JSON.parse(bytes.toString()) as Record<string, unknown>;
    syncBodies.push(body);
    if (failSync !== 'none') {
      const conflict = failSync === 'conflict';
      failSync = 'none';
      if (conflict) manifest = { ...manifest, revision: manifest.revision + 1 };
      response.statusCode = conflict ? 409 : 503;
      response.end(JSON.stringify({ detail: { code: conflict ? 'UPLOAD_REVISION_CONFLICT' : 'UPLOAD_SYNC_FAILED', message: '첨부 변경을 완료하지 못했습니다.', files: [] } }));
      return;
    }
    const additions = body.add as { upload_id: string; replace_file_id: string | null }[];
    if (syncResult !== null) {
      manifest = syncResult;
      for (const add of additions) stagedFiles.delete(add.upload_id);
      response.end(JSON.stringify(manifest));
      return;
    }
    const removed = new Set(body.remove as string[]);
    const files = body.clear ? [] : manifest.files.filter((file) => !removed.has(file.file_id) && !additions.some((add) => add.replace_file_id === file.file_id));
    for (const add of additions) {
      const staged = stagedFiles.get(add.upload_id);
      if (!staged) throw new Error('Expected a staged upload');
      files.push({ ...staged, file_id: add.replace_file_id ?? add.upload_id, source_uri: `upload:///attachments/${add.upload_id}` });
    }
    manifest = { ...manifest, revision: manifest.revision + 1, files };
    response.end(JSON.stringify(manifest));
    return;
  }
  response.statusCode = 404;
  response.end('{}');
});

beforeAll(async () => {
  // jsdom's File/FormData cannot be encoded by Node's real fetch implementation.
  // Keep both at the same native transport boundary, without replacing the client.
  const nativeForm = await new Response('', { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }).formData();
  vi.stubGlobal('FormData', nativeForm.constructor);
  vi.stubGlobal('File', buffer.File);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Expected a port');
  apiBaseUrl = `http://127.0.0.1:${address.port}`;
});
beforeEach(() => {
  manifest = { epoch: 'attachment-epoch', revision: 0, files: [] };
  stagedCount = 0;
  syncBodies = [];
  stagedFiles = new Map();
  failSync = 'none';
  syncResult = null;
  sentQueries = [];
  restoredQueries = [];
  manifestGate = null;
});
afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  vi.unstubAllGlobals();
});

function Harness({ manifestConfirmed = true, clientConfig = config }: { manifestConfirmed?: boolean; clientConfig?: ClientConfig }) {
  const [client] = useState(() => new AgentSessionClient({ sessionId: 'attachments', baseUrl: apiBaseUrl }));
  const [state, setState] = useState(createAttachmentState);
  const [confirmed, setConfirmed] = useState<UploadManifest | null>(manifest);
  const [busy, setBusy] = useState(false);
  const attachments = useAttachments({
    client, config: clientConfig, state, onChange: setState, manifest: confirmed, manifestConfirmed,
    ensureSession: () => client.refreshUploads(),
    onManifestChange: setConfirmed, onBusyChange: setBusy,
    onSendPrompt: (prompt) => sentQueries.push(prompt),
    onRestorePrompt: (prompt) => { restoredQueries.push(prompt); return true; },
    // The app's disabled flag covers other operations; this hook owns upload busy state.
    disabled: false,
  });
  return <><aside>{attachments.manifestPanel}</aside>{attachments.picker}{attachments.panel}<button disabled={busy} onClick={() => { void attachments.submit('  원문 질문\n둘째 줄  '); }}>파일과 질문 제출</button></>;
}

describe('첨부를 명시적으로 확정하는 흐름', () => {
  it('파일 추가를 여러 번 해도 기존 선택을 유지하고 함께 확정한다', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['first'], 'first.py'));
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['second'], 'second.py'));
    expect(screen.getByRole('button', { name: 'first.py 선택 취소 1' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'second.py 선택 취소 2' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    await screen.findByRole('button', { name: 'second.py 첨부 해제' });
    expect(manifest.files.map((file) => file.name)).toEqual(['first.py', 'second.py']);
    const files = within(screen.getByRole('region', { name: '함께 검색할 파일' }));
    expect(files.getAllByText(/검색 가능/)).toHaveLength(1);
    expect(files.getByText(/2개 · 11 B/)).toBeVisible();
    expect(files.getByRole('button', { name: 'first.py 첨부 해제' })).toHaveTextContent(/^해제$/);
    expect(files.getByRole('button', { name: '첨부 목록 새로고침' })).toHaveTextContent(/^새로고침$/);
    expect(files.getByRole('button', { name: '첨부 전체 해제' })).toHaveTextContent(/^전체 해제$/);
    expect(sentQueries).toHaveLength(0);
  });

  it('여러 번 추가한 선택 전체에 파일 개수 제한을 적용한다', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), [new File(['a'], 'a.py'), new File(['b'], 'b.py')]);
    await user.upload(screen.getByLabelText('파일 첨부'), [new File(['c'], 'c.py'), new File(['d'], 'd.py')]);
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    expect(await screen.findByText('한 대화에는 최대 3개 파일을 첨부할 수 있습니다.')).toBeVisible();
    expect(stagedCount).toBe(0);
    expect(syncBodies).toHaveLength(0);
    expect(manifest.files).toHaveLength(0);
  });

  it('이전 파일을 포함해 추가 선택해도 고유 파일의 용량으로 확정한다', async () => {
    const user = userEvent.setup();
    const content = 'x'.repeat(40);
    const names = ['a.py', 'b.py', 'c.py', 'd.py'];
    syncResult = { ...manifest, revision: 1, files: names.map((name, index) => ({
      file_id: `file-${index}`, name, size_bytes: 40, content_hash: hash(content), source_uri: `upload:///attachments/file-${index}`,
    })) };
    render(<Harness clientConfig={{ ...config, upload: { ...config.upload, max_files: 4 } }} />);
    await user.upload(screen.getByLabelText('파일 첨부'), names.slice(0, 3).map((name) => new File([content], name)));
    await user.upload(screen.getByLabelText('파일 첨부'), ['A.PY', ...names.slice(1)].map((name) => new File([content], name)));
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    await screen.findByRole('button', { name: 'd.py 첨부 해제' });
    expect(stagedCount).toBe(7);
    expect(syncBodies).toHaveLength(1);
    expect(stagedFiles.size).toBe(0);
    expect(sentQueries).toHaveLength(0);
  });

  it('같은 내용이라도 이름이 다르면 기존 첨부와 합쳐 용량 한도를 적용한다', async () => {
    manifest.files = [original];
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['x'.repeat(100)], 'a.py'));
    await user.upload(screen.getByLabelText('파일 첨부'), [new File(['x'.repeat(100)], 'a.py'), new File(['x'.repeat(100)], 'b.py')]);
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    expect(await screen.findByText('전체 첨부 크기 한도를 넘었습니다.')).toBeVisible();
    expect(stagedCount).toBe(0);
    expect(syncBodies).toHaveLength(0);
    expect(manifest.files).toEqual([original]);
  });

  it('이름과 크기와 수정 시각이 같아도 다른 내용은 중복으로 버리지 않는다', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['a'.repeat(100)], 'sample.py', { lastModified: 1 }));
    await user.upload(screen.getByLabelText('파일 첨부'), [
      new File(['b'.repeat(100)], 'sample.py', { lastModified: 1 }), new File(['x'], 'other.py'),
    ]);
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    expect(await screen.findByText('sample.py: 한 번에 같은 이름의 서로 다른 파일을 추가할 수 없습니다.')).toBeVisible();
    expect(stagedCount).toBe(3);
    expect(syncBodies).toHaveLength(0);
    expect(manifest.files).toHaveLength(0);
  });

  it.each([false, true])('목록 조회 중 파일 변경을 막고 응답 후 다시 허용한다 (선택 파일: %s)', async (selected) => {
    manifest.files = [original];
    const user = userEvent.setup();
    render(<Harness />);
    if (selected) await user.upload(screen.getByLabelText('파일 첨부'), new File(['next'], 'next.py'));
    let release = () => {};
    manifestGate = new Promise<void>((resolve) => { release = resolve; });
    try {
      await user.click(screen.getByRole('button', { name: '첨부 목록 새로고침' }));
      expect(screen.getByRole('button', { name: '파일 추가' })).toBeDisabled();
      expect(screen.getByLabelText('파일 첨부')).toBeDisabled();
      expect(screen.getByRole('button', { name: 'sample.py 첨부 해제' })).toBeDisabled();
      expect(screen.getByRole('button', { name: '첨부 전체 해제' })).toBeDisabled();
      if (selected) {
        expect(screen.getByRole('button', { name: 'next.py 선택 취소 1' })).toBeDisabled();
        expect(screen.queryByRole('button', { name: '첨부만 확정' })).not.toBeInTheDocument();
      }
      expect(stagedCount).toBe(0);
      expect(syncBodies).toHaveLength(0);
    } finally {
      release();
      await waitFor(() => expect(screen.getByRole('button', { name: '파일 추가' })).toBeEnabled());
    }
    if (selected) expect(screen.getByRole('button', { name: 'next.py 선택 취소 1' })).toBeEnabled();
    else expect(screen.getByRole('button', { name: 'sample.py 첨부 해제' })).toBeEnabled();
  });

  it('사이드바 목록 새로고침은 서버 목록만 갱신하고 선택 파일과 질문을 유지한다', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['next'], 'next.py'));
    manifest = { ...manifest, revision: 1, files: [original] };
    await user.click(screen.getByRole('button', { name: '첨부 목록 새로고침' }));
    expect(await screen.findByRole('button', { name: 'sample.py 첨부 해제' })).toBeVisible();
    expect(screen.getByText('next.py')).toBeVisible();
    expect(stagedCount).toBe(0);
    expect(syncBodies).toHaveLength(0);
    expect(sentQueries).toHaveLength(0);
  });

  it('확인되지 않은 캐시 목록은 검색 가능한 첨부와 구분한다', () => {
    manifest.files = [original];
    render(<Harness manifestConfirmed={false} />);
    expect(screen.getByText('이전 첨부 기록')).toBeVisible();
    expect(screen.queryByText('검색 가능')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'sample.py 첨부 해제' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '첨부 전체 해제' })).toBeDisabled();
  });

  it('파일 선택과 stage는 첨부 확정 및 질문 전송과 구분한다', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['VALUE = 42\n'], 'sample.py'));
    expect(stagedCount).toBe(0);
    expect(syncBodies).toHaveLength(0);
    expect(screen.getByText('전송 대기')).toBeVisible();
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    expect(await screen.findByRole('button', { name: 'sample.py 첨부 해제' })).toBeVisible();
    expect(manifest.files[0]?.name).toBe('sample.py');
    expect(sentQueries).toHaveLength(0);
    expect(syncBodies).toHaveLength(1);
  });

  it('다른 내용의 같은 이름은 교체 승인 후 정확한 질문을 한 번 보낸다', async () => {
    manifest.files = [original];
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['new'], 'sample.py'));
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    const confirm = await screen.findByRole('button', { name: '첨부 변경 확정' });
    expect(confirm).toBeDisabled();
    expect(syncBodies).toHaveLength(0);
    expect(manifest.files).toEqual([original]);
    await user.click(screen.getByRole('checkbox', { name: 'sample.py 교체 승인' }));
    await user.click(confirm);
    await waitFor(() => expect(sentQueries).toEqual(['  원문 질문\n둘째 줄  ']));
    expect(syncBodies[0]?.add).toEqual([expect.objectContaining({ replace_file_id: 'original' })]);
    expect(manifest.files[0]?.content_hash).toBe(hash('new'));
  });

  it.each([1, 2])('동일 내용 파일 %s개를 다시 골라도 모든 staging을 확인하고 원래 질문을 전달한다', async (count) => {
    manifest.files = [original];
    syncResult = manifest;
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), Array.from({ length: count }, () => new File(['old'], 'sample.py')));
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    await waitFor(() => expect(sentQueries).toEqual(['  원문 질문\n둘째 줄  ']));
    expect(syncBodies).toHaveLength(1);
    expect(syncBodies[0]?.add).toEqual(Array.from({ length: count }, (_, index) => ({
      upload_id: String(index + 1).padStart(32, '0'), name: 'sample.py', content_hash: hash('old'), replace_file_id: null,
    })));
    expect(stagedFiles.size).toBe(0);
  });

  it.each([false, true])('동일한 신규 파일 중복 선택도 모두 확인하고 교체 승인은 한 번만 전달한다 (기존 첨부: %s)', async (replace) => {
    if (replace) manifest.files = [original];
    syncResult = { ...manifest, revision: 1, files: [{ ...original, content_hash: hash('new') }] };
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), [new File(['new'], 'sample.py'), new File(['new'], 'sample.py')]);
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    if (replace) {
      const approvals = await screen.findAllByRole('checkbox', { name: 'sample.py 교체 승인' });
      await user.click(approvals[0]);
      await user.click(screen.getByRole('button', { name: '첨부 변경 확정' }));
    }
    await waitFor(() => expect(sentQueries).toEqual(['  원문 질문\n둘째 줄  ']));
    expect(syncBodies).toHaveLength(1);
    expect(syncBodies[0]?.add).toEqual([
      { upload_id: '1'.padStart(32, '0'), name: 'sample.py', content_hash: hash('new'), replace_file_id: replace ? 'original' : null },
      { upload_id: '2'.padStart(32, '0'), name: 'sample.py', content_hash: hash('new'), replace_file_id: null },
    ]);
    expect(stagedFiles.size).toBe(0);
  });

  it('실패한 같은 작업을 재시도해도 보류 질문은 클릭 전 보내지 않는다', async () => {
    failSync = 'once';
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['VALUE = 42'], 'sample.py'));
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    await user.click(await screen.findByRole('button', { name: '첨부 다시 시도' }));
    const send = await screen.findByRole('button', { name: '보류한 질문 보내기' });
    expect(sentQueries).toHaveLength(0);
    expect(syncBodies).toHaveLength(2);
    expect(syncBodies[1]).toEqual(syncBodies[0]);
    await user.click(send);
    expect(sentQueries).toEqual(['  원문 질문\n둘째 줄  ']);
  });

  it('revision 충돌 뒤에는 최신 목록에서 교체를 다시 승인한다', async () => {
    manifest.files = [original];
    failSync = 'conflict';
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['new'], 'sample.py'));
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    await user.click(await screen.findByRole('checkbox', { name: 'sample.py 교체 승인' }));
    await user.click(screen.getByRole('button', { name: '첨부 변경 확정' }));
    await user.click(await screen.findByRole('button', { name: '최신 첨부로 다시 검토' }));
    expect(await screen.findByRole('checkbox', { name: 'sample.py 교체 승인' })).not.toBeChecked();
    expect(screen.getByRole('button', { name: '첨부 변경 확정' })).toBeDisabled();
    await user.click(screen.getByRole('checkbox', { name: 'sample.py 교체 승인' }));
    await user.click(screen.getByRole('button', { name: '첨부 변경 확정' }));
    await screen.findByRole('button', { name: '보류한 질문 보내기' });
    expect(sentQueries).toHaveLength(0);
    expect(syncBodies[1]?.operation_id).not.toBe(syncBodies[0]?.operation_id);
    expect(syncBodies[1]?.expected_revision).toBe(1);
  });

  it('선택 오류와 복구는 질문을 보내지 않고 원문을 작성창으로 돌려준다', async () => {
    const user = userEvent.setup({ applyAccept: false });
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['bad'], 'wrong.exe'));
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    expect(await screen.findByText(/wrong.exe: 지원하지 않는 파일 형식/)).toBeVisible();
    await user.click(screen.getByRole('button', { name: '파일 다시 선택' }));
    expect(restoredQueries).toEqual(['  원문 질문\n둘째 줄  ']);
    expect(sentQueries).toHaveLength(0);
    expect(stagedCount).toBe(0);
  });

  it('같은 이름의 오류 파일만 선택 취소하고 정상 파일과 보류 질문을 유지한다', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), [new File(['good'], 'sample.py'), new File(['x'.repeat(101)], 'sample.py')]);
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    await screen.findByText('sample.py: 파일당 크기 한도를 넘었습니다.');
    expect(screen.getByRole('button', { name: 'sample.py 선택 취소 1' }).closest('li')).not.toHaveTextContent('선택 오류');
    expect(screen.getByRole('button', { name: 'sample.py 선택 취소 2' }).closest('li')).toHaveTextContent('선택 오류');
    await user.click(screen.getByRole('button', { name: 'sample.py 선택 취소 2' }));
    expect(screen.getByRole('button', { name: 'sample.py 선택 취소 1' })).toBeEnabled();
    await user.click(screen.getByRole('button', { name: '첨부 다시 시도' }));
    await screen.findByRole('button', { name: '보류한 질문 보내기' });
    expect(manifest.files).toEqual([expect.objectContaining({ name: 'sample.py', content_hash: hash('good') })]);
    expect(sentQueries).toHaveLength(0);
  });

  it('첨부 해제 충돌도 최신 목록을 검토한 뒤 명시적으로 다시 확정한다', async () => {
    manifest.files = [original];
    failSync = 'conflict';
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole('button', { name: 'sample.py 첨부 해제' }));
    await user.click(await screen.findByRole('button', { name: '최신 첨부로 다시 검토' }));
    expect(manifest.files).toEqual([original]);
    await user.click(await screen.findByRole('button', { name: '첨부 변경 확정' }));
    await waitFor(() => expect(manifest.files).toEqual([]));
    expect(syncBodies[1]?.operation_id).not.toBe(syncBodies[0]?.operation_id);
    expect(syncBodies[1]?.remove).toEqual(['original']);
    expect(sentQueries).toHaveLength(0);
  });
});
