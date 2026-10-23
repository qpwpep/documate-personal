from types import SimpleNamespace

import pytest

from src.core.planner_schema import RetrievalTask
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag import build_upload_search_tool


def test_empty_call_notation_retrieves_the_same_uploaded_definition():
    source = 'def check_exists(path):\n    return path.exists()\n'
    parsed = chunk_python_text(
        path='check_paths.py', text=source, chunk_size=800, chunk_overlap=120,
    ).parsed
    retriever = SimpleNamespace(source_documents=(parsed,))
    search = build_upload_search_tool()
    results = []
    tasks = []

    for symbol in ('check_exists', 'check_exists()'):
        task = RetrievalTask(
            route='upload', query=f'{symbol} 함수 원문', k=1,
            requirement={'symbols': [symbol], 'match': 'definition'},
        )
        tasks.append(task)
        results.append(search(query=task.query, requirement=task.requirement, retriever=retriever))

    assert [result.diagnostics.answerability for result in results] == ['covered', 'covered']
    assert [result.hits[0].evidence.excerpt for result in results] == [source.rstrip('\n')] * 2
    assert results[0].hits[0].evidence.id == results[1].hits[0].evidence.id
    assert tasks[0].requirement_id == tasks[1].requirement_id


@pytest.mark.parametrize('match', ['symbol', 'definition'])
def test_empty_call_notation_preserves_qualified_definition_and_file_scope(match):
    source = ('class Paths:\n    def check_exists(self):\n        return True\n\n'
              'class Other:\n    def check_exists(self):\n        return False\n')
    parsed = chunk_python_text(
        path='check_paths.py', text=source, chunk_size=800, chunk_overlap=120,
    ).parsed
    parsed.elements[0].metadata['file_id'] = 'known-file'
    retriever = SimpleNamespace(source_documents=(parsed,))
    search = build_upload_search_tool()
    task = RetrievalTask(
        route='upload', query='Paths.check_exists() 함수 원문', k=1,
        requirement={'symbols': ['Paths.check_exists()'], 'match': match, 'file_ids': ['known-file']},
    )

    result = search(query=task.query, requirement=task.requirement, retriever=retriever)

    assert result.diagnostics.answerability == 'covered'
    assert [hit.evidence.excerpt for hit in result.hits] == [
        '    def check_exists(self):\n        return True',
    ]
    unknown_file = RetrievalTask(
        route='upload', query=task.query, k=1,
        requirement={'symbols': ['Paths.check_exists()'], 'match': match, 'file_ids': ['unknown-file']},
    )
    unavailable = search(query=unknown_file.query, requirement=unknown_file.requirement, retriever=retriever)
    assert unavailable.hits == []
    assert unavailable.diagnostics.answerability == 'unknown'


@pytest.mark.parametrize('symbol', [
    'check_exists(path)', 'check_exists(value=path)', 'factory().check_exists()',
    'check_exists()()', 'Unknown.check_exists()', 'Paths..check_exists()',
])
def test_call_expressions_and_unknown_qualifiers_cannot_expand_to_another_symbol(symbol):
    source = 'def check_exists(path):\n    return path.exists()\n'
    parsed = chunk_python_text(
        path='check_paths.py', text=source, chunk_size=800, chunk_overlap=120,
    ).parsed
    task = RetrievalTask(
        route='upload', query=f'{symbol} 함수 원문', k=1,
        requirement={'symbols': [symbol], 'match': 'definition'},
    )

    result = build_upload_search_tool()(
        query=task.query, requirement=task.requirement,
        retriever=SimpleNamespace(source_documents=(parsed,)),
    )

    assert result.hits == []
    assert result.diagnostics.answerability == 'missing'


@pytest.mark.parametrize(('route', 'match'), [('docs', 'symbol'), ('upload', 'topic')])
def test_non_code_lookup_and_literal_aspects_preserve_original_notation(route, match):
    task = RetrievalTask(
        route=route, query='check_exists()의 의미', k=1,
        requirement={'symbols': ['check_exists()'], 'match': match, 'aspects': ['check_exists()']},
    )

    assert task.requirement.symbols == ['check_exists()']
    assert task.requirement.aspects == ['check_exists()']


def test_code_symbol_normalization_does_not_change_literal_aspects():
    task = RetrievalTask(
        route='upload', query='check_exists() 함수 안의 Path(path).exists() 원문', k=1,
        requirement={'symbols': ['check_exists()'], 'match': 'definition',
                     'aspects': ['Path(path).exists()']},
    )

    assert task.requirement.symbols == ['check_exists']
    assert task.requirement.aspects == ['Path(path).exists()']
