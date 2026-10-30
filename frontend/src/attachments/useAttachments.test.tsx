import { createServer } from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import * as buffer from 'node:buffer';
import { useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
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
let sentQueries: string[] = [];
let restoredQueries: string[] = [];
let apiBaseUrl = '';

const server = createServer(async (request, response) => {
  response.setHeader('Content-Type', 'application/json');
  if (request.method === 'GET' && request.url?.endsWith('/uploads')) {
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
  sentQueries = [];
  restoredQueries = [];
});
afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  vi.unstubAllGlobals();
});

function Harness({ manifestConfirmed = true }: { manifestConfirmed?: boolean }) {
  const [client] = useState(() => new AgentSessionClient({ sessionId: 'attachments', baseUrl: apiBaseUrl }));
  const [state, setState] = useState(createAttachmentState);
  const [confirmed, setConfirmed] = useState<UploadManifest | null>(manifest);
  const [busy, setBusy] = useState(false);
  const attachments = useAttachments({
    client, config, state, onChange: setState, manifest: confirmed, manifestConfirmed,
    ensureSession: () => client.refreshUploads(),
    onManifestChange: setConfirmed, onBusyChange: setBusy,
    onSendPrompt: (prompt) => sentQueries.push(prompt),
    onRestorePrompt: (prompt) => restoredQueries.push(prompt),
    disabled: busy,
  });
  return <>{attachments.panel}<button disabled={busy} onClick={() => { void attachments.submit('  원문 질문\n둘째 줄  '); }}>파일과 질문 제출</button></>;
}

describe('첨부를 명시적으로 확정하는 흐름', () => {
  it('확인되지 않은 캐시 목록은 검색 가능한 첨부와 구분한다', () => {
    manifest.files = [original];
    render(<Harness manifestConfirmed={false} />);
    expect(screen.getByText('이전 첨부 기록')).toBeVisible();
    expect(screen.queryByText('검색 가능한 첨부')).not.toBeInTheDocument();
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
    expect(await screen.findByText('검색 가능한 첨부')).toBeVisible();
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

  it('동일 내용은 확정 revision을 바꾸지 않는다', async () => {
    manifest.files = [original];
    const user = userEvent.setup();
    render(<Harness />);
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['old'], 'sample.py'));
    await user.click(screen.getByRole('button', { name: '파일과 질문 제출' }));
    await waitFor(() => expect(sentQueries).toEqual(['  원문 질문\n둘째 줄  ']));
    expect(syncBodies).toHaveLength(0);
    expect(manifest.revision).toBe(0);
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
