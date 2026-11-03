import { Fragment, createElement, useId, useState, type ReactNode } from 'react';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import { unified } from 'unified';
import type { components } from '../api/generated';
import type { AgentResponse, AnswerResponse } from '../api/types';
import { EvidencePanel, type AnswerContext } from './EvidencePanel';
import { MarkdownView } from './MarkdownView';
import './answer.css';

type AnswerBlock = AnswerResponse['content']['blocks'][number];
type ContentUnit = components['schemas']['ContentUnit'];
type Citation = components['schemas']['Citation'];
type Problem = components['schemas']['ExecutionProblem'];
type Unit = { path: string; unit: ContentUnit; isCode?: boolean; inline?: boolean };
type OpenCitation = (citation: Citation, opener: HTMLElement) => void;

const basisLabels = { source: '', interaction: '', inference: '해석', example: '예시', excerpt: '원문 발췌' };
const markdownParser = unified().use(remarkParse).use(remarkGfm);
const actionLabels = { save_text: '파일 저장', slack_notify: 'Slack 전송' };
const actionStatuses = { success: '완료', error: '실패', skipped: '실행하지 않음', unknown: '결과 확인 필요' };
const deliveryGuidance = {
  correct_input: '보낼 대상 하나의 ID 또는 이메일을 확인해 주세요.',
  fix_configuration: '앱 권한과 설정을 확인한 뒤 같은 수신자로 재시도해 주세요.',
  retry_same_target: '같은 수신자로 재시도할 수 있습니다.',
  verify_delivery: 'Slack에서 전달 여부를 확인해 주세요. 자동으로 다시 전송하지 않습니다.',
};

export function AnswerView({ response, turn, actions, requestId }: {
  response: AnswerResponse; turn?: AgentResponse; actions?: ReactNode; requestId?: string | null;
}) {
  const answerId = useId();
  const [opened, setOpened] = useState<{ citation: Citation; context: AnswerContext[]; opener: HTMLElement } | null>(null);
  const citations = new Map(response.citations.map((citation) => [citation.evidence.id, citation]));
  const units = response.content.blocks.flatMap((block, index) => blockUnits(block, index));
  const usedRefs = [...new Set(units.flatMap(({ unit }) => unit.refs))];
  const limits = validationLimits(response);
  const issueCodes = new Set(response.issues.map((issue) => issue.code));
  const globalIssues = response.issues.filter((issue) => !issue.unit_id);
  const partialProblem = turn?.status === 'partial' && turn.problem?.code !== 'save_failed'
    && turn.problem?.stage !== 'action_postprocess' ? turn.problem : null;
  const sections: { blocks: { block: AnswerBlock; index: number }[]; units: Unit[] }[] = [];
  for (const [index, block] of response.content.blocks.entries()) {
    if (!sections.length || (block.type === 'heading' && sections[sections.length - 1].blocks.length)) {
      sections.push({ blocks: [], units: [] });
    }
    sections[sections.length - 1].blocks.push({ block, index });
    sections[sections.length - 1].units.push(...blockUnits(block, index));
  }

  function openCitation(citation: Citation, opener: HTMLElement, contextUnits: Unit[]) {
    setOpened({ citation, opener,
      context: contextUnits.filter(({ unit }) => unit.refs.includes(citation.evidence.id))
        .map(({ unit, isCode, inline }) => ({ text: unit.text, isCode: isCode || unit.source_location != null, inline })),
    });
  }

  return (
    <div className="answer-view" id={answerId}>
      {partialProblem && <PartialNotice response={response} problem={partialProblem} />}
      {!partialProblem && turn?.problem && (
        <div className="answer-notices">
          <p>{turn.problem.message}</p>
          {problemGuidance(turn.problem) && <p>{problemGuidance(turn.problem)}</p>}
        </div>
      )}
      <div className="answer-body">
        {sections.map((section, sectionIndex) => {
          return (
            <Fragment key={sectionIndex}>
              {section.blocks.map(({ block, index }) => {
                const blockContent = blockUnits(block, index);
                const paths = new Set(blockContent.map(({ path }) => path));
                const localIssues = response.issues.filter((issue) => issue.unit_id && paths.has(issue.unit_id));
                return (
                  <div className="answer-block" key={index}>
                    <BodyBlock block={block} citations={citations} onCitation={(citation, opener) => openCitation(citation, opener, section.units)} />
                    {localIssues.length > 0 && (
                      <details className="answer-limit-details">
                        <summary>이 내용의 한계 ({localIssues.length}개)</summary>
                        {localIssues.map((issue, issueIndex) => (
                          <div key={issueIndex}>
                            <p className="answer-caption">관련 내용: {blockContent.find(({ path }) => path === issue.unit_id)?.unit.text}</p>
                            <p>{issue.message}</p>
                          </div>
                        ))}
                      </details>
                    )}
                    {response.checks.filter((check) => paths.has(check.unit_id)).map((check) => (
                      check.reference_status === 'missing'
                        ? <p className="answer-integrity-warning" key={check.unit_id}>이 내용에 연결된 근거를 찾지 못했습니다.</p>
                        : check.support_status === 'unsupported'
                          ? <p className="answer-integrity-warning" key={check.unit_id}>이 인용문은 보관한 원문과 일치하지 않습니다.</p>
                          : null
                    ))}
                  </div>
                );
              })}
            </Fragment>
          );
        })}
      </div>
      {(globalIssues.length > 0 || partialProblem) && (
        <details className="answer-limit-details">
          <summary>답변의 한계와 처리 정보</summary>
          {globalIssues.map((issue, index) => <p key={index}>{issue.message}</p>)}
          {partialProblem && turn?.request_id && <p className="answer-caption">오류 식별자: {turn.request_id}</p>}
        </details>
      )}
      {!partialProblem && limits.length > 0 && <div className="answer-validation-notices">
        <p className="answer-caption">확인할 한계: {limits.join(' ')}</p>
        {issueCodes.has('invalid_content_removed')
          ? <p className="answer-caption">필요한 항목이 빠졌다면 해당 항목을 지정해 다시 질문해 주세요.</p>
          : issueCodes.has('source_location_unverified')
            ? <p className="answer-caption">출처에서 원문 위치를 확인하거나, 위치를 확인할 수 있는 원본 파일을 첨부해 주세요.</p>
            : null}
      </div>}
      <div className="answer-toolbar">
        {usedRefs.map((ref) => {
          const citation = citations.get(ref);
          if (!citation) return null;
          const title = citation.evidence.snapshot.title || citation.evidence.snapshot.source_uri;
          return <button type="button" className="answer-source" key={ref} aria-label={`근거 [${citation.number}] ${title}`}
            onClick={(event) => openCitation(citation, event.currentTarget, units)}>
            <small aria-hidden="true">{citation.number}</small><span>{title}</span>
          </button>;
        })}
        {actions}
        <ValidationSummary response={response} units={units} limits={limits} incomplete={turn?.status === 'partial'} requestId={requestId} />
      </div>
      <ActionReceipts response={response} />
      {opened && <EvidencePanel {...opened} onClose={() => setOpened(null)} />}
    </div>
  );
}

function blockUnits(block: AnswerBlock, index: number): Unit[] {
  const prefix = `b${index}`;
  switch (block.type) {
    case 'paragraph': return block.content.map((unit, item) => ({ path: `${prefix}.content.${item}`, unit }));
    case 'list': return block.items.map((unit, item) => ({ path: `${prefix}.items.${item}`, unit }));
    case 'heading': return [{ path: `${prefix}.content`, unit: block.content }];
    case 'code': return [{ path: `${prefix}.content`, unit: block.content, isCode: true, inline: block.inline }];
    case 'table': return [
      ...block.columns.map((unit, column) => ({ path: `${prefix}.columns.${column}`, unit })),
      ...block.rows.flatMap((row, rowIndex) => row.map((unit, column) => ({ path: `${prefix}.rows.${rowIndex}.${column}`, unit }))),
    ];
  }
}

function BodyBlock({ block, citations, onCitation }: { block: AnswerBlock; citations: Map<string, Citation>; onCitation: OpenCitation }) {
  const renderUnit = (unit: ContentUnit) => <UnitText unit={unit} citations={citations} onCitation={onCitation} />;
  switch (block.type) {
    case 'paragraph': {
      // Exact input copies keep the whole Markdown document in one paragraph unit.
      const content = block.content.map((unit) => ({ unit, inline: unit.source_location != null || !hasBlockMarkdown(unit.text) }));
      const Paragraph = content.every(({ inline }) => inline) ? 'p' : 'div';
      return <Paragraph>{content.map(({ unit, inline }, index) => <Fragment key={index}>
        {index > 0 && ' '}<UnitText unit={unit} citations={citations} onCitation={onCitation} inline={inline} />
      </Fragment>)}</Paragraph>;
    }
    case 'heading':
      return createElement(`h${block.level}`, {}, renderUnit(block.content));
    case 'list': {
      const List = block.ordered ? 'ol' : 'ul';
      return <List>{block.items.map((unit, index) => <li key={index}>{renderUnit(unit)}</li>)}</List>;
    }
    case 'code': {
      const basis = basisLabels[block.content.basis];
      const label = block.content.basis === 'excerpt' ? '원문 발췌'
        : `${basis ? `${basis} · ` : ''}${block.inline ? '인라인 코드' : '코드 블록'} · 실행 확인 안 됨`;
      return <>
        {block.inline ? <p><code>{block.content.text}</code></p>
          : <AnswerCode text={block.content.text} language={block.language || '코드'} />}
        <p className="answer-caption">{label} <CitationLinks unit={block.content} citations={citations} onCitation={onCitation} /></p>
      </>;
    }
    case 'table':
      return <div className="answer-table"><table>
        <thead><tr>{block.columns.map((unit, column) => <th scope="col" key={column}>{renderUnit(unit)}</th>)}</tr></thead>
        <tbody>{block.rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((unit, column) => <td key={column}>{renderUnit(unit)}</td>)}</tr>)}</tbody>
      </table></div>;
  }
}

function AnswerCode({ text, language }: { text: string; language: string }) {
  const [copyStatus, setCopyStatus] = useState<'copied' | 'failed' | null>(null);

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus('copied');
    } catch {
      setCopyStatus('failed');
    }
  }

  return <div className="answer-code">
    <div className="answer-code-heading"><span>{language}</span>
      <button type="button" className="answer-code-copy secondary-button" aria-label="코드 복사" onClick={() => void copyCode()}>
        {copyStatus === 'copied' ? '복사 완료' : '코드 복사'}
      </button>
    </div>
    <pre><code>{text}</code></pre>
    <p className={`answer-code-copy-status${copyStatus !== 'failed' ? ' visually-hidden' : ''}`} role="status" aria-label="코드 복사 상태">
      {copyStatus === 'copied' && '코드를 복사했습니다.'}
      {copyStatus === 'failed' && '복사하지 못했습니다. 코드를 선택해 복사하거나 다시 시도해 주세요.'}
    </p>
  </div>;
}

function hasBlockMarkdown(text: string): boolean {
  return markdownParser.parse(text).children.some((node) => node.type !== 'paragraph' && node.type !== 'definition');
}

function UnitText({ unit, citations, onCitation, inline = true }: {
  unit: ContentUnit; citations: Map<string, Citation>; onCitation: OpenCitation; inline?: boolean;
}) {
  const basis = basisLabels[unit.basis];
  const Container = inline ? 'span' : 'div';
  return <Container className={inline ? 'answer-unit' : 'answer-markdown-unit'}>
    {basis && <span className="answer-basis">{basis} · </span>}
    {unit.source_location ? <span>{unit.text}</span> : <MarkdownView text={unit.text} inline={inline} />}
    <CitationLinks unit={unit} citations={citations} onCitation={onCitation} />
  </Container>;
}

function CitationLinks({ unit, citations, onCitation }: { unit: ContentUnit; citations: Map<string, Citation>; onCitation: OpenCitation }) {
  return unit.refs.map((ref) => {
    const citation = citations.get(ref);
    return citation && <button className="answer-inline-citation" type="button" key={ref}
      aria-label={`근거 [${citation.number}] 열기`} onClick={(event) => onCitation(citation, event.currentTarget)}>[{citation.number}]</button>;
  });
}

function PartialNotice({ response, problem }: { response: AnswerResponse; problem: Problem }) {
  const globalIssues = response.issues.filter((issue) => !issue.unit_id);
  const codes = new Set(globalIssues.map((issue) => issue.code));
  const provided = globalIssues.find((issue) => issue.code === 'source_excerpt_fallback')?.message ?? '현재 제공할 수 있는 본문을 아래에 표시합니다.';
  const reasons = [...new Set(globalIssues.filter((issue) => !['answer_incomplete', 'source_excerpt_fallback'].includes(issue.code)).map((issue) => issue.message))];
  let guidance = problemGuidance(problem);
  if (problem.next_action === 'supply_information') {
    if (codes.has('evidence_packet_incomplete') && !codes.has('retrieved_evidence_incomplete')) {
      guidance = '질문을 항목별로 나누어 다시 요청해 주세요. 검색한 자료를 답변에 담는 과정에서 생긴 한계입니다.';
    } else if (codes.has('source_location_unverified')) {
      guidance = '출처에서 원문 위치를 확인하거나, 위치를 확인할 수 있는 원본 파일을 첨부해 주세요.';
    } else if (codes.has('invalid_content_removed')) {
      guidance = '필요한 항목이 빠졌다면 해당 항목을 지정해 다시 질문해 주세요.';
    } else {
      guidance = '위 조건을 확인해 질문을 수정하세요. 필요한 내용이 기존 자료에 있으면 파일명·위치를 지정하고, 없는 경우에만 자료를 추가해 주세요.';
    }
  }
  return <section className="answer-partial-notice" aria-label="부분 답변">
    <p><strong>부분 답변</strong> · 요청 전체를 완성하지 못했습니다.</p>
    <p>{problem.message}</p>
    <p><strong>제공한 범위</strong></p><p>{provided}</p>
    {reasons.length > 0 && <><p><strong>확인하지 못한 내용</strong></p><ul>{reasons.slice(0, 2).map((reason) => <li key={reason}>{reason}</li>)}</ul></>}
    {reasons.length > 2 && <p>나머지 {reasons.length - 2}개 한계는 ‘답변의 한계와 처리 정보’에서 확인하세요.</p>}
    {codes.has('invalid_content_removed') && <p>제외된 항목의 상세 기록은 없습니다.</p>}
    {guidance && <><p><strong>다음 행동</strong></p><p>{guidance}</p></>}
  </section>;
}

function problemGuidance(problem: Problem): string {
  if (problem.retry_after_seconds != null) return `${problem.retry_after_seconds}초 후 다시 시도할 수 있습니다.`;
  return { none: '', retry_later: '잠시 후 같은 요청으로 다시 시도할 수 있습니다.',
    fix_configuration: '서비스 설정을 수정해야 합니다. 오류 식별자를 운영자에게 알려 주세요.',
    supply_information: '확인에 필요한 자료나 조건을 추가해 주세요.' }[problem.next_action];
}

function validationLimits(response: AnswerResponse): string[] {
  const report = response.semantic_validation;
  const requestCheck = report?.checks.find((check) => check.name === 'request_fulfilled' && !check.unit_id);
  const limits = [...new Set(response.issues.filter((issue) => !issue.unit_id).map((issue) => issue.message))];
  if (response.issues.some((issue) => issue.code === 'invalid_content_removed')) limits.push('제외된 항목의 상세 기록은 없습니다.');
  const unitLimitCount = response.issues.filter((issue) => issue.unit_id).length;
  if (unitLimitCount) limits.push(`내용별 한계 ${unitLimitCount}개는 해당 내용 아래에서 확인할 수 있습니다.`);
  if (requestCheck?.status === 'refused') limits.push('요청 조건을 자동으로 확인하지 못했습니다.');
  else if (requestCheck?.probability != null && report && requestCheck.probability < report.threshold) limits.push('답변 본문의 요청 조건 충족을 확인하지 못했습니다.');
  return limits;
}

function ValidationSummary({ response, units, limits, incomplete, requestId }: {
  response: AnswerResponse; units: Unit[]; limits: string[]; incomplete: boolean; requestId?: string | null;
}) {
  const report = response.semantic_validation;
  if (!response.citations.length && !report && !units.length && !requestId) return null;
  const resolved = response.checks.filter((check) => check.reference_status === 'resolved').length;
  const excerpts = new Set(units.filter(({ unit }) => unit.basis === 'excerpt').map(({ path }) => path));
  const exact = response.checks.filter((check) => excerpts.has(check.unit_id) && check.support_status === 'exact_match').length;
  const quoteSummary = excerpts.size ? ` · 원문 발췌 ${excerpts.size}개 중 원문 일치 ${exact}개` : '';
  const issueCodes = new Set(response.issues.map((issue) => issue.code));
  const requestCheck = report?.checks.find((check) => check.name === 'request_fulfilled' && !check.unit_id);
  return <details className="answer-validation"><summary>답변 정보</summary>
    <p className="answer-caption">출처가 연결된 내용 {resolved}개{quoteSummary}</p>
    {requestId && <p className="request-id">요청 ID: {requestId}</p>}
      <p>{resolved ? `출처 연결: 내용 ${resolved}개에 출처가 연결되어 있습니다.` : '출처 연결: 이 답변에 연결된 출처가 없습니다.'}</p>
      <p>{excerpts.size ? `원문 발췌 ${excerpts.size}개 중 원문 일치 ${exact}개. 보관한 원문과 일치하는지 확인합니다.` : '원문 발췌: 자료에서 그대로 옮긴 내용은 없습니다.'}</p>
      {units.some(({ isCode }) => isCode) && <p>실행 확인: 코드 형식으로 표시된 내용의 실행 결과는 확인하지 않았습니다.</p>}
      {limits.length > 0 && <p>확인하지 못한 내용: {limits.join(' ')}</p>}
      <p>자동 확인은 답변 전체의 정확도나 실행 성공을 보증하지 않습니다.</p>
      <p>본문 확인과 저장·전송의 처리 결과는 별도로 표시됩니다.</p>
      {['retrieved_evidence_incomplete', 'evidence_packet_incomplete', 'answer_incomplete', 'source_excerpt_fallback'].some((code) => issueCodes.has(code)) && <p>이 답변에 기록된 한계가 있습니다. 아래 판정은 제공된 본문 범위에 한정됩니다.</p>}
      {incomplete && <p>이번 요청 처리는 완료되지 않았습니다. 남은 작업은 안내와 저장·전송 결과를 확인해 주세요.</p>}
      <details><summary>기술 상세</summary>
        {report && <p>수치는 개별 조건의 충족 여부에 대한 자동 평가입니다.</p>}
        <SemanticResult label="답변 본문 조건 충족" check={requestCheck} threshold={report?.threshold} />
        {report?.checks.filter((check) => check.name !== 'request_fulfilled' && check.name !== 'grounded').map((check, index) => <SemanticResult key={index}
          label={check.name === 'format_fulfilled' ? '답변 서식 조건 충족' : check.name}
          check={check} threshold={report.threshold} />)}
        {units.map(({ path, unit }, index) => <section key={path} aria-label={`내용 ${index + 1} 검증`}>
          <p><strong>내용 {index + 1}</strong> · {unit.text}</p>
          <SemanticResult label="근거 지지" check={report?.checks.find((check) => check.name === 'grounded' && check.unit_id === path)} threshold={report?.threshold} />
        </section>)}
      </details>
  </details>;
}

function SemanticResult({ label, check, threshold }: { label: string; check?: components['schemas']['SemanticCheck']; threshold?: number }) {
  if (!check || threshold == null) return <p>{label}: 검증 안 됨</p>;
  if (check.status === 'refused') return <p>{label}: 평가 거절 · 충족 확률 없음</p>;
  return <p>{label}: {check.probability != null && check.probability >= threshold ? '통과' : '미통과'} · 충족 확률 {check.probability} · 임곗값 {threshold}</p>;
}

function ActionReceipts({ response }: { response: AnswerResponse }) {
  if (!response.actions.length) return null;
  return <section className="answer-actions" aria-label="전달 결과"><h3>전달 결과</h3>
    {response.actions.map((action, index) => {
      const delivery = action.slack;
      const label = delivery?.status === 'unknown' ? '전달 여부 불명' : actionStatuses[action.status];
      return <div key={index}>
        <p><strong>{actionLabels[action.kind]} · {label}</strong></p>
        {delivery?.selection && <p>{delivery.selection.source === 'configured_default' ? '기본 수신자' : '요청 수신자'}: {delivery.selection.selector.value}</p>}
        {delivery?.intent.state === 'unresolved' && <p>요청 수신자: {delivery.intent.raw_input}</p>}
        {delivery?.status === 'sent' && <p>전송 대상: {delivery.target?.channel_id} · 메시지 식별자: {delivery.message_ts}</p>}
        {[...new Set([action.message, action.error, delivery?.failure?.message])].filter((message): message is string => Boolean(message)).map((message) => <p key={message}>{message}</p>)}
        {delivery?.failure && <p>{delivery.failure.retry_after_seconds != null && `${delivery.failure.retry_after_seconds}초 이후 `}{deliveryGuidance[delivery.failure.next_action]}</p>}
        {action.kind === 'save_text' && action.status === 'error' && <p>기존 답변은 유지됩니다. 같은 답변으로 저장을 다시 시도할 수 있습니다.</p>}
      </div>;
    })}
  </section>;
}
