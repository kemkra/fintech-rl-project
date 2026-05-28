import argparse
import ast
import json
from pathlib import Path
import py_compile
import textwrap
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_TOOLS_FILE = PROJECT_ROOT / "src" / "tools" / "generated_market_tools.py"
GENERATED_SCHEMAS_FILE = PROJECT_ROOT / "src" / "llm" / "generated_tool_schemas.py"
TOOLS_START = "# <generated-tools>"
TOOLS_END = "# </generated-tools>"
SCHEMAS_START = "# <generated-schemas>"
SCHEMAS_END = "# </generated-schemas>"


class PromotionError(ValueError):
    pass


def slugify_tool_name(text):
    text = str(text or "new_tool").strip().lower()
    chars = []
    previous_underscore = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
        elif not previous_underscore:
            chars.append("_")
            previous_underscore = True
    return "".join(chars).strip("_") or "new_tool"


def strip_code_fence(code):
    code = str(code or "").strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return textwrap.dedent(code).strip() + "\n"


def find_proposal_file(proposal_id, search_roots):
    candidates = []
    for root in search_roots:
        root = Path(root)
        if root.exists():
            candidates.extend(root.glob(f"**/{proposal_id}.json"))
    if not candidates:
        raise PromotionError(f"Proposal id not found: {proposal_id}")
    if len(candidates) > 1:
        joined = "\n".join(str(path) for path in candidates)
        raise PromotionError(f"Multiple proposal files matched {proposal_id}:\n{joined}")
    return candidates[0]


def load_proposal(args):
    if args.proposal_file:
        path = Path(args.proposal_file)
    else:
        search_roots = [
            PROJECT_ROOT / ".streamlit_runtime",
            PROJECT_ROOT / "reports" / "tool_proposals",
        ]
        path = find_proposal_file(args.proposal_id, search_roots)

    if not path.exists():
        raise PromotionError(f"Proposal file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        proposal = json.load(file)
    proposal["_proposal_file"] = str(path)
    return proposal


def validate_function_code(code, tool_name):
    code = strip_code_fence(code)
    if not code.strip():
        raise PromotionError("Proposal does not include suggested_python_code.")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PromotionError(f"Suggested Python code has a syntax error: {exc}") from exc

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise PromotionError("Suggested Python code must contain exactly one top-level function.")
    function = functions[0]
    if function.name != tool_name:
        raise PromotionError(f"Function name {function.name!r} does not match promoted tool name {tool_name!r}.")

    allowed_top_level = (ast.FunctionDef,)
    for node in tree.body:
        if not isinstance(node, allowed_top_level):
            raise PromotionError("Suggested Python code may only contain one function definition. Use generated file common imports.")

    blocked_names = {"eval", "exec", "compile", "open", "input", "__import__"}
    blocked_modules = {"os", "subprocess", "socket", "shutil"}
    blocked_attrs = {"system", "popen", "remove", "unlink", "rmdir", "rmtree"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PromotionError("Imports are not allowed inside promoted proposal code.")
        if isinstance(node, ast.Name) and node.id in blocked_names:
            raise PromotionError(f"Blocked unsafe name in proposal code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in blocked_attrs:
            raise PromotionError(f"Blocked unsafe attribute in proposal code: {node.attr}")
        if isinstance(node, ast.Name) and node.id in blocked_modules:
            raise PromotionError(f"Blocked unsafe module reference in proposal code: {node.id}")

    return code


def build_schema(proposal, tool_name):
    schema = proposal.get("suggested_tool_schema") or {}
    if schema.get("type") == "function" and "function" in schema:
        schema["function"]["name"] = tool_name
        return schema
    if schema.get("name"):
        schema["name"] = tool_name
        return {"type": "function", "function": schema}

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": proposal.get("user_need") or f"Generated tool: {tool_name}",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def insert_block(path, start_marker, end_marker, block, duplicate_key):
    text = path.read_text(encoding="utf-8")
    if duplicate_key in text:
        raise PromotionError(f"{duplicate_key} already exists in {path}.")
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    new_text = text[:start] + "\n" + block.rstrip() + "\n" + text[end:]
    path.write_text(new_text, encoding="utf-8")


def promote(proposal):
    tool_name = slugify_tool_name(proposal.get("tool_name") or proposal.get("display_name"))
    proposal_id = proposal.get("proposal_id") or tool_name
    code = validate_function_code(proposal.get("suggested_python_code"), tool_name)
    schema = build_schema(proposal, tool_name)

    write = bool(proposal.get("write_tool", False))
    chat_workspace = proposal.get("chat_workspace_tool", True)

    tool_block = (
        f"# Generated from proposal: {proposal_id}\n"
        f"@register_generated_tool({tool_name!r}, write={write!r}, chat_workspace={bool(chat_workspace)!r})\n"
        f"{code}"
    )
    schema_block = (
        f"# Generated from proposal: {proposal_id}\n"
        f"GENERATED_TOOL_SCHEMAS.append({json.dumps(schema, ensure_ascii=False, indent=4)})\n"
    )

    insert_block(GENERATED_TOOLS_FILE, TOOLS_START, TOOLS_END, tool_block, f"@register_generated_tool({tool_name!r}")
    insert_block(GENERATED_SCHEMAS_FILE, SCHEMAS_START, SCHEMAS_END, schema_block, f'"name": "{tool_name}"')

    proposal_file = Path(proposal["_proposal_file"])
    proposal["status"] = "promoted"
    proposal["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    proposal["promoted_to"] = {
        "tools_file": str(GENERATED_TOOLS_FILE),
        "schemas_file": str(GENERATED_SCHEMAS_FILE),
    }
    proposal_file.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    py_compile.compile(str(GENERATED_TOOLS_FILE), doraise=True)
    py_compile.compile(str(GENERATED_SCHEMAS_FILE), doraise=True)
    py_compile.compile(str(PROJECT_ROOT / "src" / "llm" / "assistant.py"), doraise=True)

    return tool_name


def parse_args():
    parser = argparse.ArgumentParser(description="Promote a reviewed Tool Proposal into generated tool files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proposal-file", help="Path to a proposal JSON file.")
    source.add_argument("--proposal-id", help="Proposal id to find under runtime/report proposal directories.")
    return parser.parse_args()


def main():
    args = parse_args()
    proposal = load_proposal(args)
    tool_name = promote(proposal)
    print(f"Promoted tool: {tool_name}")
    print(f"Updated: {GENERATED_TOOLS_FILE}")
    print(f"Updated: {GENERATED_SCHEMAS_FILE}")


if __name__ == "__main__":
    main()
