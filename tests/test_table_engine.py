import io
import json
from dataclasses import dataclass

import pandas as pd
import pytest
from openpyxl import Workbook

from src.table_engine import (
    MAX_RESULT_ROWS,
    QueryPlan,
    TablePlanError,
    TableQAService,
    execute_plan,
    frame_to_text,
    is_table_file,
    load_tables,
    looks_like_table_question,
    parse_plan,
    sensitive_column_values,
)


def make_workbook() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append([None, None, None])
    sheet.append(["Region", "Sales Rep", "Amount"])
    rows = [
        ("South", "Arun Kumar", "₹1,200"),
        ("South", "Priya Sharma", 800),
        ("North", "Arun Kumar", "500"),
        ("East", "Ravi", "(100)"),
    ]
    for row in rows:
        sheet.append(list(row))
    workbook.create_sheet("Notes").append(["only header"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def tables() -> dict[str, pd.DataFrame]:
    return load_tables("sales.xlsx", make_workbook())


def test_load_tables_uses_first_non_empty_row_and_parses_currency(tables) -> None:
    frame = tables["Sales"]

    assert list(frame.columns) == ["Region", "Sales Rep", "Amount"]
    assert pd.api.types.is_numeric_dtype(frame["Amount"])
    assert frame["Amount"].tolist() == [1200, 800, 500, -100]
    assert "Notes" not in tables


def test_load_csv_with_semicolons() -> None:
    data = b"city;visits\nChennai;10\nCoimbatore;5\n"

    frame = load_tables("visits.csv", data)["CSV"]

    assert frame["visits"].sum() == 15


def test_non_table_files_return_nothing() -> None:
    assert load_tables("notes.pdf", b"%PDF") == {}
    assert is_table_file("A.XLSX") and not is_table_file("a.docx")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the total amount for South?", True),
        ("How many rows have Region North?", True),
        ("Average sales per region", True),
        ("What does this document say about refunds?", False),
    ],
)
def test_router(question: str, expected: bool) -> None:
    assert looks_like_table_question(question) is expected


def test_filter_and_sum_uses_every_row(tables) -> None:
    plan = QueryPlan(
        sheet="sales",
        filters=[{"column": "region", "op": "eq", "value": "SOUTH"}],
        aggregations=[{"column": "Amount", "func": "sum"}],
    )

    result = execute_plan(tables, plan)

    assert result.to_dict("records") == [{"sum_Amount": 2000}]
    assert plan.sheet == "Sales"


def test_group_by_sort_and_limit(tables) -> None:
    plan = QueryPlan(
        sheet="Sales",
        group_by=["Region"],
        aggregations=[
            {"column": "Amount", "func": "sum"},
            {"column": "Sales Rep", "func": "count"},
        ],
        sort={"column": "sum_Amount", "descending": True},
        limit=2,
    )

    result = execute_plan(tables, plan)

    assert result["Region"].tolist() == ["South", "North"]
    assert result["count_Sales Rep"].tolist() == [2, 1]


def test_numeric_comparison_and_column_selection(tables) -> None:
    plan = QueryPlan(
        sheet="Sales",
        filters=[{"column": "Amount", "op": "ge", "value": "800"}],
        columns=["Sales Rep"],
    )

    assert execute_plan(tables, plan)["Sales Rep"].tolist() == [
        "Arun Kumar",
        "Priya Sharma",
    ]


def test_contains_filter(tables) -> None:
    plan = QueryPlan(
        sheet="Sales",
        filters=[{"column": "Sales Rep", "op": "contains", "value": "kumar"}],
        aggregations=[{"column": "Amount", "func": "mean"}],
    )

    assert execute_plan(tables, plan)["mean_Amount"].tolist() == [850]


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (QueryPlan(sheet="Missing"), "Sheet 'Missing' does not exist"),
        (
            QueryPlan(sheet="Sales", columns=["Profit"]),
            "Column 'Profit' does not exist",
        ),
        (
            QueryPlan(
                sheet="Sales", aggregations=[{"column": "Region", "func": "sum"}]
            ),
            "not numeric",
        ),
        (
            QueryPlan(
                sheet="Sales",
                filters=[{"column": "Amount", "op": "gt", "value": "lots"}],
            ),
            "not a number",
        ),
    ],
)
def test_invalid_plans_raise_clear_errors(
    tables, plan: QueryPlan, message: str
) -> None:
    with pytest.raises(TablePlanError, match=message):
        execute_plan(tables, plan)


def test_parse_plan_extracts_json_and_clamps_limit() -> None:
    plan = parse_plan('Sure! {"sheet": "Sales", "limit": 5000} Done.')

    assert plan is not None
    assert plan.limit == MAX_RESULT_ROWS


def test_parse_plan_returns_none_for_unanswerable() -> None:
    assert parse_plan('{"sheet": ""}') is None


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("no json here", "did not return"),
        ("{not json}", "not valid JSON"),
        ('{"sheet": "Sales", "filters": [{"column": "A", "op": "drop"}]}', "invalid"),
    ],
)
def test_parse_plan_rejects_bad_replies(reply: str, message: str) -> None:
    with pytest.raises(TablePlanError, match=message):
        parse_plan(reply)


def test_frame_to_text_formats_numbers_and_truncates() -> None:
    frame = pd.DataFrame({"n": [1234.5, 2000.0, None]})

    assert frame_to_text(frame) == "n\n1,234.5\n2,000\n"
    assert frame_to_text(frame, max_rows=1).endswith("... 2 more rows")


def test_sensitive_column_values(tables) -> None:
    values = sensitive_column_values(
        tables
        | {
            "Other": pd.DataFrame(
                {"Product name": ["Pen"], "Customer Address": ["x y z"]}
            )
        }
    )

    assert values["PERSON"] == ["Arun Kumar", "Priya Sharma", "Ravi"]
    assert values["ADDRESS"] == ["x y z"]


@dataclass
class FakeResponse:
    content: str


class ScriptedModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.replies[len(self.prompts) - 1])


def test_service_masks_names_and_restores_filter_values(tables) -> None:
    plan = {
        "sheet": "Sales",
        "filters": [{"column": "Sales Rep", "op": "eq", "value": "<PERSON_1>"}],
        "aggregations": [{"column": "Amount", "func": "sum"}],
    }
    model = ScriptedModel(
        [json.dumps(plan), "<PERSON_1> sold 1,700 in total on sheet Sales."]
    )
    service = TableQAService(tables, model, redact_personal_data=True)

    answer = service.answer("What is the total amount for Arun Kumar?")

    assert answer is not None
    assert answer.result.to_dict("records") == [{"sum_Amount": 1700}]
    assert answer.answer == "Arun Kumar sold 1,700 in total on sheet Sales."
    assert all("Arun Kumar" not in prompt for prompt in model.prompts)
    assert "Priya Sharma" not in model.prompts[0]
    assert '<computed_result sheet="Sales">\nsum_Amount\n1,700' in model.prompts[1]
    assert answer.masked_counts["PERSON"] >= 1


def test_service_returns_none_when_model_says_not_answerable(tables) -> None:
    model = ScriptedModel(['{"sheet": ""}'])

    assert TableQAService(tables, model).answer("Summarise the notes") is None
    assert len(model.prompts) == 1


def test_service_requires_tables_and_question(tables) -> None:
    with pytest.raises(ValueError, match="At least one table"):
        TableQAService({}, ScriptedModel([]))

    with pytest.raises(ValueError, match="cannot be empty"):
        TableQAService(tables, ScriptedModel([])).answer("  ")


def test_service_masked_entities_limits_what_is_hidden(tables) -> None:
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "sheet": "Sales",
                    "aggregations": [{"column": "Amount", "func": "sum"}],
                }
            ),
            "Total is 2,400.",
        ]
    )
    service = TableQAService(
        tables,
        model,
        redact_personal_data=True,
        masked_entities=frozenset({"EMAIL"}),
    )

    answer = service.answer("total amount")

    assert answer is not None
    assert "Arun Kumar" in model.prompts[0]
    assert answer.masked_counts == {}
