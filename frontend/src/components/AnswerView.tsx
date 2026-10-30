import { createElement } from 'react';
import type { AnswerResponse } from '../api/types';

const actionLabels = { save_text: '파일 저장', slack_notify: 'Slack 전송' };
const actionStatuses = { success: '완료', error: '실패', skipped: '실행하지 않음', unknown: '결과 확인 필요' };

export function AnswerView({ response }: { response: AnswerResponse }) {
  return (
    <div className="answer-view">
      {response.issues.length > 0 && (
        <section className="answer-notices" aria-label="답변의 한계와 안내">
          {response.issues.map((issue, index) => <p key={index}>{issue.message}</p>)}
        </section>
      )}
      {response.checks.some((check) => check.reference_status === 'missing' || check.support_status === 'unsupported') && (
        <p className="answer-notices">일부 내용의 원문 연결을 확인하지 못했습니다. 출처와 답변의 한계를 함께 확인해 주세요.</p>
      )}
      <div className="answer-body">
        {response.content.blocks.map((block, index) => {
          switch (block.type) {
            case 'paragraph':
              return <p key={index}>{block.content.map((unit) => unit.text).join(' ')}</p>;
            case 'heading':
              return createElement(`h${block.level ?? 2}`, { key: index }, block.content.text);
            case 'list': {
              const List = block.ordered ? 'ol' : 'ul';
              return <List key={index}>{block.items.map((unit, item) => <li key={item}>{unit.text}</li>)}</List>;
            }
            case 'code':
              return block.inline
                ? <p key={index}><code>{block.content.text}</code></p>
                : <pre key={index}><code>{block.content.text}</code></pre>;
            case 'table':
              return (
                <div className="answer-table" key={index}>
                  <table>
                    <thead><tr>{block.columns.map((unit, column) => <th scope="col" key={column}>{unit.text}</th>)}</tr></thead>
                    <tbody>{block.rows?.map((row, rowIndex) => (
                      <tr key={rowIndex}>{row.map((unit, column) => <td key={column}>{unit.text}</td>)}</tr>
                    ))}</tbody>
                  </table>
                </div>
              );
          }
        })}
      </div>
      {response.citations.length > 0 && (
        <details className="answer-sources">
          <summary>출처 {response.citations.length}개</summary>
          {response.citations.map(({ number, evidence }) => (
            <section key={evidence.id}>
              <h3>[{number}] {evidence.snapshot.title || evidence.snapshot.source_uri}</h3>
              {/^https?:\/\//i.test(evidence.snapshot.source_uri) && (
                <a href={evidence.snapshot.source_uri} target="_blank" rel="noreferrer">현재 출처 페이지 열기</a>
              )}
              <p>보관된 원문 요소</p>
              <pre>{evidence.element.text || evidence.element.table?.cells?.map((cell) => cell.text).join('\n')}</pre>
              {evidence.snapshot.capture_scope === 'provider_excerpt' && <p>전체 문서가 아닌 일부 발췌를 수집했습니다.</p>}
              {evidence.snapshot.quality_issues?.map((issue, index) => <p key={index}>{issue}</p>)}
            </section>
          ))}
        </details>
      )}
      {response.actions.length > 0 && (
        <section className="answer-actions" aria-label="전달 결과">
          <h3>전달 결과</h3>
          {response.actions.map((action, index) => (
            <div key={index}>
              <p><strong>{actionLabels[action.kind]} · {actionStatuses[action.status]}</strong></p>
              {[...new Set([action.message, action.error, action.slack?.failure?.message])]
                .filter((message): message is string => Boolean(message))
                .map((message) => <p key={message}>{message}</p>)}
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
