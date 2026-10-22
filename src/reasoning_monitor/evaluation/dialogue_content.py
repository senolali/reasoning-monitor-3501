"""Conservative numeric-equation and template-specific transition diagnostics."""

import ast
import operator
import re
from fractions import Fraction

import pandas as pd

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


def numeric_expression(text):
    text = text.strip().rstrip('.').replace(',', '')
    if not text or len(text) > 200 or not re.fullmatch(r'[\d\s.+*/()\-]+', text):
        raise ValueError('unsupported expression')
    tree = ast.parse(text, mode='eval')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        raise ValueError('unsupported syntax')

    return visit(tree.body)


def equations(raw):
    checked = []
    for line in raw.splitlines():
        line = line.replace('**', '').replace('×', '*').replace('−', '-')
        line = re.sub(r'^\s*\d+\.\s+', '', line)
        line = re.sub(r'^\s*[-*]\s+', '', line)
        line = re.sub(r'\([^()]*[a-zA-Z][^()]*\)', '', line)
        parts = line.split('=')
        for left, right in zip(parts, parts[1:], strict=False):
            left = left.rsplit(':', 1)[-1].strip()
            right = right.strip()
            try:
                lhs, rhs = numeric_expression(left), numeric_expression(right)
            except (ValueError, SyntaxError, ZeroDivisionError):
                continue
            checked.append(dict(left=left, right=right, wrong=lhs != rhs))
    return checked


def prompt_delta(template, prompt):
    patterns = {
        'research_budget': [
            (r'sponsor has added (\d+)', 1),
            (r'refund of (\d+)', 1),
            (r'expenditure of (\d+)', -1),
        ],
        'study_cohort': [
            (r'enrolled (\d+) additional', 1),
            (r'count downward by (\d+)', 1),
            (r'recorded (\d+) additional', -1),
        ],
    }
    if template == 'software_incident':
        match = re.search(r'(\d+) (?:additional )?(critical|major|minor) incident', prompt)
        if match:
            return int(match[1]) * {'critical': 5, 'major': 2, 'minor': 1}[match[2]] * (
                -1 if 'resolved' in prompt else 1
            )
    for pattern, sign in patterns.get(template, []):
        match = re.search(pattern, prompt, flags=re.I)
        if match:
            return sign * int(match[1])
    return None


def integer_answer(value):
    text = str(value).strip().rstrip('.').replace(',', '')
    return int(text) if re.fullmatch(r'[-+]?\d+', text) else None


def content_frame(records):
    rows, audit = [], []
    previous = {}
    for record in sorted(records, key=lambda r: (r['dialogue_id'], r['condition'], r['turn'])):
        key = (record['dialogue_id'], record['condition'])
        tests = equations(record['raw_text'])
        wrong = sum(test['wrong'] for test in tests)
        answer = integer_answer(record['answer'])
        delta = prompt_delta(record['template'], record['user_prompt'])
        prior = previous.get(key)
        available = record['turn'] > 1 and prior is not None and answer is not None and delta is not None
        rows.append(dict(
            qid=record['qid'], condition=record['condition'],
            equation_count=len(tests), arithmetic_available=int(bool(tests)),
            arithmetic_error_rate=wrong / len(tests) if tests else 0.0,
            arithmetic_any_error=int(wrong > 0),
            transition_available=int(available),
            transition_violation=int(available and answer != prior + delta),
        ))
        for test in tests:
            audit.append(dict(qid=record['qid'], condition=record['condition'], **test))
        previous[key] = answer
    return pd.DataFrame(rows), audit
