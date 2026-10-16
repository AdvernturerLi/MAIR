"""Guard the approved data boundary: validation pixels stay on local services."""
import ast
from pathlib import Path


def test_distillation_calls_external_llm_with_statistics_text_only():
    source=Path(__file__).parents[1]/'pilot.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='gpt']
    assert len(calls)==1
    assert calls[0].args==[]
    assert {kw.arg for kw in calls[0].keywords}=={'prompt'}
    assert isinstance(calls[0].keywords[0].value,ast.Name) and calls[0].keywords[0].value.id=='prompt'


def test_depictqa_evaluation_uses_loopback_service():
    base=Path(__file__).parents[2]
    source=next(p for p in (base/'AgenticIR/llm/depictqa.py',base/'agenticir_additions/llm/depictqa.py') if p.is_file())
    tree=ast.parse(source.read_text(encoding='utf-8'))
    urls=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('http')]
    assert urls and all(u.startswith('http://127.0.0.1:') for u in urls)


def test_external_distillation_is_guarded_and_opt_in():
    tree=ast.parse((Path(__file__).parents[1]/'pilot.py').read_text(encoding='utf-8'))
    run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run')
    all_calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='distill']
    guarded=[]
    for node in ast.walk(run):
        if isinstance(node,ast.If) and ast.unparse(node.test)=='self.args.distill':
            guarded.extend(n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='distill')
    assert len(all_calls)==len(guarded)==2
    assert any(isinstance(n,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='--distill' for a in n.args)
               and any(k.arg=='action' and isinstance(k.value,ast.Constant) and k.value.value=='store_true' for k in n.keywords)
               for n in ast.walk(tree))


def test_pairwise_external_call_remains_text_only():
    tree=ast.parse((Path(__file__).parents[1]/'distill_pairwise_queue.py').read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='gpt']
    assert len(calls)==1 and not calls[0].args
    assert {k.arg for k in calls[0].keywords}=={'prompt'}
