import glob
import json
import math
import os
import re

import gradio as gr
import pandas as pd
from black import FileMode, format_str

from sliders.log_utils import logger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def list_incorrect_or_unknown(filter_choice: str):
    """List buckets that are Incorrect or Unknown based on filter_choice in {All, Incorrect, Unknown}."""
    if not viewer.buckets:
        return "❌ Error: No prompt logs loaded"
    return viewer.render_incorrect_or_unknown_buckets(filter_choice)


# def extract_action_columns(raw: str) -> List[Dict[str, Any]]:
#     """
#     Convert a multiline string that looks like a repr of tuples into a list of
#     dicts containing only the true data ("action") columns.

#     An "action" column is defined as:
#       • present in the header tuple
#       • does *not* contain "_citation" or "_rationale"

#     Example
#     -------
#     >>> text = \"\"\"\
#     ('chunk_id', 'id', 'wages_expense_FY2023_citation', 'wages_expense_FY2023_rationale',
#      'wages_expense_FY2023', 'net_sales_FY2023_citation', 'net_sales_FY2023_rationale',
#      'net_sales_FY2023')
#     (0, 0, '...', '...', 2395.3, '...', '...', 10208.6)
#     (4, 1, '...', '...', 0.0,   '...', '...', 10208580.0)
#     \"\"\"
#     >>> extract_action_columns(text)
#     [{'wages_expense_FY2023': 2395.3, 'net_sales_FY2023': 10208.6},
#      {'wages_expense_FY2023': 0.0,    'net_sales_FY2023': 10208580.0}]
#     """
#     # Split non-blank lines, preserve order
#     lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]

#     # First line is the header tuple → literal Python tuple
#     header: tuple = ast.literal_eval(lines[0])

#     # Decide which columns we want
#     keep_idx = [
#         i
#         for i, col in enumerate(header)
#         if not col.endswith(("_quote", "_rationale"))
#         # Optionally keep only those that *look* like fiscal-year data
#         # e.g. if you know you only want *_FY2023*
#         # and col.lower().endswith("fy2023")
#     ]
#     keep_names = [header[i] for i in keep_idx]

#     rows: List[Dict[str, Any]] = []
#     safe_globals = {"__builtins__": {}}
#     safe_locals = {"list": list}
#     for ln in lines[1:]:
#         ln = ln.replace(", nan,", " None,")
#         record = eval(ln, safe_globals, safe_locals)
#         rows.append({name: record[i] for name, i in zip(keep_names, keep_idx)})

#     return rows


def extract_action_columns(raw):
    # 1) Keep only tuple lines (skip the header if you don't want it as a row)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("(")]

    # If the first tuple is the header, keep it separately:
    header = eval(lines[0], {"__builtins__": {}})  # header is safe (no nan/list)
    row_lines = lines[1:]  # the data rows

    # 2) OPTIONAL: strip list([...]) -> [...] so this also works with literal_eval later
    row_lines = [re.sub(r"\blist\(\s*(\[[\s\S]*?\])\s*\)", r"\1", ln) for ln in row_lines]

    # 3) Wrap rows in a single list expression
    expr = "[" + ",\n".join(row_lines) + "]"

    # 4) Eval with a restricted env that defines nan (and list if you didn’t strip it)
    env = {"__builtins__": {}, "nan": math.nan, "list": list}
    rows = eval(expr, env, {})  # rows is a list of tuples

    # 5) Turn into list of dicts if useful:
    records = [dict(zip(header, row)) for row in rows]

    return records


class PromptLogViewer:
    def __init__(self):
        self.prompts = []
        self.buckets = []
        self.results_data = None
        self.results_lookup = {}
        self.log_file_path = None
        self.results_file_path = None

    def load_prompts(self, log_file_path):
        """Load prompts from the JSONL file"""
        try:
            self.log_file_path = log_file_path
            with open(log_file_path, "r") as f:
                self.prompts = [json.loads(line) for line in f]
            print(f"Loaded {len(self.prompts)} prompts from {log_file_path}")
            self.create_buckets()
            return f"✅ Successfully loaded {len(self.prompts)} prompts and created {len(self.buckets)} buckets"
        except Exception as e:
            print(f"Error loading prompts: {e}")
            self.prompts = []
            self.buckets = []
            return f"❌ Error loading prompts: {str(e)}"

    def load_results(self, results_file_path):
        """Load results file to get correctness and gold answers"""
        try:
            self.results_file_path = results_file_path
            with open(results_file_path, "r") as f:
                self.results_data = json.load(f)

            # Create lookup for faster access
            self.results_lookup = []

            if isinstance(self.results_data, list):
                self.results_data = {"results": self.results_data[0]}

            for i, result in enumerate(self.results_data["results"]):
                self.results_lookup.append(result)

            print(f"Loaded {len(self.results_lookup)} results from {results_file_path}")
            return f"✅ Successfully loaded {len(self.results_lookup)} results"
        except Exception as e:
            print(f"Error loading results: {e}")
            self.results_data = None
            self.results_lookup = {}
            return f"❌ Error loading results: {str(e)}"

    def create_buckets(self):
        """Group prompts into buckets based on 'generate_schema_qa.prompt' template"""
        buckets = []
        new_bucket = []

        for prompt in self.prompts:
            if prompt.get("template_name") == "regenerate_descriptions.prompt":
                if new_bucket:  # Only add non-empty buckets
                    buckets.append(new_bucket)
                new_bucket = []
            new_bucket.append(prompt)

        if new_bucket:  # Add the last bucket
            buckets.append(new_bucket)

        # Remove the first empty bucket if it exists
        if buckets and not buckets[0]:
            buckets = buckets[1:]

        self.buckets = buckets
        print(f"Created {len(self.buckets)} buckets")

    def get_bucket_result_info(self, bucket_idx):
        """Get result information for a bucket"""
        if not self.results_lookup or bucket_idx >= len(self.buckets):
            return None

        bucket = self.buckets[bucket_idx]
        for i, prompt in enumerate(bucket):
            if prompt.get("template_name") == "rephrase_question.prompt":
                prompt_output = prompt.get("input", "")

        for res in self.results_lookup:
            if res.get("question") in prompt_output:
                return res

    def extract_question_from_bucket(self, bucket):
        """Extract question from bucket"""
        for prompt in bucket:
            if prompt.get("template_name") == "generate_schema_qa.prompt":
                input_text = prompt.get("input", "")
                if "question:" in input_text.lower():
                    lines = input_text.split("\n")
                    for line in lines:
                        if line.lower().startswith("question:"):
                            return line.split(":", 1)[1].strip()
        return "Question not found"

    def format_prompt_display(self, prompt):
        """Format a single prompt for display"""
        template_name = prompt.get("template_name", "Unknown")

        # Create more informative summary based on template type
        summary_text = f"🔧 Template: {template_name}"

        # Add extra info for specific templates
        if template_name == "generate_schema_qa.prompt":
            summary_text += " (Schema Generation)"
        elif template_name == "extract_schema.prompt":
            summary_text += " (Data Extraction)"
        elif template_name == "schema_merging_sql.prompt":
            summary_text += " (SQL Merging)"
        elif template_name.startswith("objectives_merging_sql_"):  # objectives_merging_sql_{objective_key}.prompt
            summary_text += f" (Objectives Merging SQL {template_name.split('_')[-1].split('.')[0]})"
        elif template_name.startswith("answer_"):
            summary_text += " (Answer Generation)"

        # Create a formatted display wrapped in an accordion (collapsible details)
        display = f"<details>\n<summary><strong>{summary_text}</strong></summary>\n\n"

        if "instruction" in prompt:
            display += "<details>\n<summary><strong>📋 Instruction</strong></summary>\n\n"
            display += f"```\n{prompt['instruction']}\n```\n\n"
            display += "</details>\n\n"

        if "input" in prompt:
            input_text = prompt["input"]

            # Special handling for schema_merging_sql.prompt template
            if (
                template_name == "schema_merging_sql.prompt"
                or template_name.startswith("objectives_merging_sql_")
                or template_name == "check_objective_necessity.prompt"
            ):
                display += "<details>\n<summary><strong>📥 Input</strong></summary>\n\n"

                # Try to extract and display table data as DataFrame
                try:
                    # Look for tuple data in the input
                    raw_lines = ""
                    for line in input_text.split("\n"):
                        if line.strip().startswith("("):
                            raw_lines += line + "\n"

                    if raw_lines.strip():
                        # Extract action columns and create DataFrame
                        table_data = extract_action_columns(raw_lines)
                        if table_data:
                            df = pd.DataFrame(table_data)
                            df = df.drop(
                                columns=[
                                    col for col in df.columns if col.endswith("_quote") or col.endswith("rationale")
                                ]
                            )
                            display += "**📊 Extracted Table Data:**\n"
                            table_html = df.to_html(index=False).replace("<table", "<table style='white-space: nowrap'")
                            display += f"\n<div style='overflow-x: auto; max-width: 100%;'>{table_html}</div>\n\n"

                            # Also show DataFrame info
                            display += f"**📋 Table Info:** {len(df)} rows × {len(df.columns)} columns\n\n"

                        # Show the raw input as well (full content)
                        display += f"**📄 Full Input:**\n```\n{input_text}\n```\n\n"
                    else:
                        # No table data found, show input normally
                        display += f"```\n{input_text}\n```\n\n"

                except Exception as e:
                    # If extraction fails, show input normally
                    display += f"```\n{input_text}\n```\n\n"
                    display += f"*Note: Could not parse table data: {str(e)}*\n\n"

                display += "</details>\n\n"
            else:
                # Normal handling for other templates
                display += "<details>\n<summary><strong>📥 Input</strong></summary>\n\n"
                display += f"```\n{input_text}\n```\n\n"
                display += "</details>\n\n"

        if "output" in prompt:
            output_text = prompt["output"]

            display += "<details>\n<summary><strong>📤 Output</strong></summary>\n\n"

            # Try to format output nicely
            try:
                if (
                    output_text.startswith("Classes(")
                    or output_text.startswith("ExtractionOutput(")
                    or output_text.startswith("Output(")
                ):
                    # Format structured outputs
                    pretty_code = format_str(output_text, mode=FileMode())
                    display += f"```python\n{pretty_code}\n```\n\n"
                else:
                    # Show full output content
                    display += f"```\n{output_text}\n```\n\n"
            except Exception as e:
                print(f"Error formatting output: {e}")
                display += f"```\n{output_text}\n```\n\n"

            display += "</details>\n\n"

        # Close the details accordion
        display += "</details>\n\n"

        return display

    def get_bucket_info(self, bucket_idx):
        """Get information about a specific bucket"""
        if not self.buckets or bucket_idx < 0 or bucket_idx >= len(self.buckets):
            return "❌ Invalid bucket number", "", ""

        bucket = self.buckets[bucket_idx]

        # Get result info if available
        result_info = self.get_bucket_result_info(bucket_idx)

        # Get bucket summary
        template_counts = {}
        for prompt in bucket:
            template = prompt.get("template_name", "Unknown")
            template_counts[template] = template_counts.get(template, 0) + 1

        # Determine correctness status
        status_emoji = "❓"
        status_text = "Unknown"
        if result_info:
            # Check different correctness fields
            for eval_tool_name, eval_tool_result in result_info["evaluation_tools"].items():
                if "soft" in eval_tool_name:
                    if eval_tool_result["correct"] is True:
                        status_emoji = "✅"
                        status_text = "Correct"
                    else:
                        status_emoji = "❌"
                        status_text = "Incorrect"

        summary = f"## 📊 Bucket {bucket_idx} Summary {status_emoji}\n\n"
        summary += f"**Status:** {status_text}\n\n"
        summary += f"**Total Prompts:** {len(bucket)}\n\n"
        summary += "**Template Distribution:**\n"
        for template, count in template_counts.items():
            summary += f"- `{template}`: {count}\n"

        # Extract question
        question = self.extract_question_from_bucket(bucket)
        summary += f"\n**Question:** {question}\n"

        # Add result information if available
        if result_info:
            summary += "\n**📋 Result Information:**\n"

            # Show ID if available
            for id_field in ["financebench_id", "id", "question_id"]:
                if id_field in result_info:
                    summary += f"- **ID:** `{result_info[id_field]}`\n"
                    break

            # Show predicted answer
            for answer_field in ["predicted_answer", "answer", "model_answer", "prediction"]:
                if answer_field in result_info:
                    summary += f"- **Predicted Answer:** {result_info[answer_field]}\n"
                    break

            # Show gold answer
            for gold_field in ["gold_answer", "ground_truth", "correct_answer", "expected_answer"]:
                if gold_field in result_info:
                    summary += f"- **Gold Answer:** {result_info[gold_field]}\n"
                    break

            # Show additional metrics if available
            if "score" in result_info:
                summary += f"- **Score:** {result_info['score']}\n"

        # Format all prompts
        all_prompts = ""
        for i, prompt in enumerate(bucket):
            all_prompts += f"\n---\n## 📝 Prompt {i + 1}/{len(bucket)}\n\n"
            all_prompts += self.format_prompt_display(prompt)

        return (
            summary,
            all_prompts,
            f"Showing bucket {bucket_idx} with {len(bucket)} prompts - {status_text}",
        )

    def search_buckets_by_question(self, search_term):
        """Search for buckets containing a specific question or term"""
        if not search_term.strip():
            return "Please enter a search term"

        matching_buckets = []
        for i, bucket in enumerate(self.buckets):
            question = self.extract_question_from_bucket(bucket)
            if search_term.lower() in question.lower():
                result_info = self.get_bucket_result_info(i)
                status = "❓"
                if result_info:
                    for correct_field in ["soft_correct", "correct", "is_correct"]:
                        if correct_field in result_info:
                            status = "✅" if result_info[correct_field] else "❌"
                            break
                matching_buckets.append((i, question[:200] + "...", status))

        if not matching_buckets:
            return f"No buckets found containing '{search_term}'"

        result = f"Found {len(matching_buckets)} buckets containing '{search_term}':\n\n"
        for bucket_idx, preview, status in matching_buckets[:20]:  # Show first 20 matches
            result += f"**Bucket {bucket_idx}** {status}: {preview}\n\n"

        if len(matching_buckets) > 20:
            result += f"... and {len(matching_buckets) - 20} more matches"

        return result

    def get_statistics(self):
        """Get overall statistics"""
        if not self.results_lookup:
            return "No results loaded"

        correct_count = 0
        incorrect_count = 0
        failed_count = 0

        wrong_bucket_ids = []

        for i in range(len(self.buckets)):
            result_info = self.get_bucket_result_info(i)
            if result_info:
                for eval_tool_name, eval_tool_result in result_info["evaluation_tools"].items():
                    if "soft" in eval_tool_name:
                        if eval_tool_result["correct"]:
                            correct_count += 1
                            break
                        else:
                            incorrect_count += 1
                            wrong_bucket_ids.append(i)
                            break
            else:
                failed_count += 1
                wrong_bucket_ids.append(i)

        total = len(self.buckets)
        accuracy = correct_count / total * 100 if total > 0 else 0

        stats = f"""
## 📈 Overall Statistics

**Total Buckets:** {total}

**Correct:** ✅ {correct_count} ({correct_count / total * 100:.1f}%)

**Incorrect:** ❌ {incorrect_count} ({incorrect_count / total * 100:.1f}%)

**Failed:** ❌ {failed_count} ({failed_count / total * 100:.1f}%)

**Wrong Bucket IDs:** `{wrong_bucket_ids}`

**Overall Accuracy:** {accuracy:.1f}%
"""

        return stats

    def get_bucket_status(self, bucket_idx):
        """Return status text and emoji for a given bucket.

        Status is determined as:
        - Correct: any evaluation tool marks correct=True
        - Incorrect: result exists and none of the tools mark correct=True
        - Unknown: no matching result found
        """
        result_info = self.get_bucket_result_info(bucket_idx)
        if not result_info:
            return ("Unknown", "❓")

        try:
            for eval_tool_name, eval_tool_result in result_info["evaluation_tools"].items():
                if "soft" in eval_tool_name:
                    if eval_tool_result["correct"]:
                        return ("Correct", "✅")
                    else:
                        return ("Incorrect", "❌")
            # If none marked correct=True
            return ("Incorrect", "❌")
        except Exception:
            # On any parsing error, consider status unknown
            return ("Unknown", "❓")

    def get_incorrect_or_unknown_buckets(self, filter_choice: str):
        """Return list of tuples (bucket_idx, question_preview, status_text, status_emoji)
        filtered by filter_choice in {All, Incorrect, Unknown}.
        'All' includes only Incorrect and Unknown buckets (excludes Correct).
        """
        matches = []
        for i in range(len(self.buckets)):
            status_text, status_emoji = self.get_bucket_status(i)

            if filter_choice == "Incorrect" and status_text != "Incorrect":
                continue
            if filter_choice == "Unknown" and status_text != "Unknown":
                continue
            if filter_choice == "All" and status_text == "Correct":
                continue

            question = self.extract_question_from_bucket(self.buckets[i])
            preview = question[:200] + ("..." if len(question) > 200 else "")
            matches.append((i, preview, status_text, status_emoji))

        return matches

    def render_incorrect_or_unknown_buckets(self, filter_choice: str):
        """Render a markdown list of filtered buckets based on filter_choice."""
        matches = self.get_incorrect_or_unknown_buckets(filter_choice)
        if not matches:
            return f"No buckets found for filter '{filter_choice}'"

        out = f"Found {len(matches)} buckets for filter '{filter_choice}':\n\n"
        # Limit display to first 200 entries for readability
        max_display = 200
        for i, (bucket_idx, preview, status_text, status_emoji) in enumerate(matches[:max_display]):
            out += f"**Bucket {bucket_idx}** {status_emoji} {status_text}: {preview}\n\n"

        if len(matches) > max_display:
            out += f"... and {len(matches) - max_display} more"

        return out


# Directory scanning functions
def get_prompt_log_files():
    """Get list of prompt log files from finance_bench_logs directory"""
    try:
        log_dir = os.path.join(CURRENT_DIR, "..", "logs/prompt_logs")
        if not os.path.exists(log_dir):
            return ["No logs directory found"]

        json_files = glob.glob(os.path.join(log_dir, "*.jsonl"))
        if not json_files:
            return ["No JSON files found in finance_bench_logs"]

        # Return just filenames for display
        return [os.path.basename(f) for f in sorted(json_files)]
    except Exception as e:
        return [f"Error scanning directory: {str(e)}"]


def get_results_files():
    """Get list of result files from results directory"""
    try:
        results_dir = os.path.join(CURRENT_DIR, "..", "results")
        if not os.path.exists(results_dir):
            return ["No results directory found"]

        json_files = glob.glob(os.path.join(results_dir, "*.json"))
        if not json_files:
            return ["No JSON files found in results"]

        # Return just filenames for display
        return [os.path.basename(f) for f in sorted(json_files)]
    except Exception as e:
        return [f"Error scanning directory: {str(e)}"]


# Initialize the viewer
viewer = PromptLogViewer()


def load_prompt_file_dropdown(filename):
    """Load prompt log file from dropdown selection"""
    if not filename or filename.startswith("No ") or filename.startswith("Error"):
        return "❌ No valid file selected"

    try:
        log_dir = os.path.join(CURRENT_DIR, "..", "logs/prompt_logs")
        full_path = os.path.join(log_dir, filename)
        return viewer.load_prompts(full_path)
    except Exception as e:
        return f"❌ Error loading prompt log file: {str(e)}"


def load_results_file_dropdown(filename):
    """Load results file from dropdown selection"""
    if not filename or filename.startswith("No ") or filename.startswith("Error"):
        return "❌ No valid file selected"

    try:
        results_dir = os.path.join(CURRENT_DIR, "..", "results")
        full_path = os.path.join(results_dir, filename)
        return viewer.load_results(full_path)
    except Exception as e:
        return f"❌ Error loading results file : {str(e)}"


def view_bucket(bucket_number):
    """View a specific bucket"""
    if not viewer.buckets:
        return "❌ Error: No prompt logs loaded", "", "No data loaded"

    try:
        bucket_idx = int(bucket_number)
        return viewer.get_bucket_info(bucket_idx)
    except ValueError:
        return "❌ Please enter a valid bucket number", "", "Invalid input"
    except Exception as e:
        logger.error(f"Error while viewing bucket: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return f"❌ Error while viewing bucket: {str(e)}", "", "Error occurred"


def search_questions(search_term):
    """Search for questions containing specific terms"""
    if not viewer.buckets:
        return "❌ Error while searching questions: No prompt logs loaded"

    return viewer.search_buckets_by_question(search_term)


def get_stats():
    """Get statistics"""
    if not viewer.buckets:
        return "No data loaded"
    return viewer.get_statistics()


# Create Gradio interface
with gr.Blocks(title="Enhanced Prompt Log Viewer", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔍 Enhanced Prompt Log Viewer")
    gr.Markdown("Navigate through prompt logs with correctness tagging and gold answers")

    with gr.Tab("📁 Load Files"):
        gr.Markdown("## Load Data Files")
        gr.Markdown("Select files from the available directories:")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Prompt Log Files")
                prompt_dropdown = gr.Dropdown(
                    label="Select Prompt Log File", choices=get_prompt_log_files(), value=None, interactive=True
                )
                load_prompts_btn = gr.Button("Load Selected Prompt File", variant="primary")
                prompt_status = gr.Textbox(label="Prompt Load Status", interactive=False)
                refresh_prompts_btn = gr.Button("🔄 Refresh File List", variant="secondary", size="sm")

            with gr.Column():
                gr.Markdown("### 📊 Results Files")
                results_dropdown = gr.Dropdown(
                    label="Select Results File", choices=get_results_files(), value=None, interactive=True
                )
                load_results_btn = gr.Button("Load Selected Results File", variant="primary")
                results_status = gr.Textbox(label="Results Load Status", interactive=False)
                refresh_results_btn = gr.Button("🔄 Refresh File List", variant="secondary", size="sm")

        # Event handlers
        load_prompts_btn.click(fn=load_prompt_file_dropdown, inputs=[prompt_dropdown], outputs=[prompt_status])

        load_results_btn.click(fn=load_results_file_dropdown, inputs=[results_dropdown], outputs=[results_status])

        # Refresh buttons
        refresh_prompts_btn.click(
            fn=lambda: gr.Dropdown.update(choices=get_prompt_log_files()), outputs=[prompt_dropdown]
        )

        refresh_results_btn.click(
            fn=lambda: gr.Dropdown.update(choices=get_results_files()), outputs=[results_dropdown]
        )

    with gr.Tab("📊 Statistics"):
        stats_button = gr.Button("🔄 Refresh Statistics", variant="primary")
        stats_output = gr.Markdown(label="Statistics")

        stats_button.click(fn=get_stats, outputs=[stats_output])

    with gr.Tab("📂 Browse by Bucket"):
        with gr.Row():
            with gr.Column(scale=1):
                bucket_input = gr.Number(label="Bucket Number", value=0, minimum=0, step=1)
                view_button = gr.Button("🔍 View Bucket", variant="primary")

            with gr.Column(scale=3):
                status_output = gr.Textbox(label="Status", interactive=False, max_lines=1)

        summary_output = gr.Markdown(label="Bucket Summary")
        prompts_output = gr.Markdown(label="All Prompts")

        view_button.click(
            fn=view_bucket, inputs=[bucket_input], outputs=[summary_output, prompts_output, status_output]
        )

    with gr.Tab("🔎 Search Questions"):
        with gr.Row():
            search_input = gr.Textbox(
                label="Search Term",
                placeholder="Enter keywords to search in questions (e.g., 'CVS Health', 'gross margin')",
                lines=1,
            )
            search_button = gr.Button("🔍 Search", variant="primary")

        search_output = gr.Markdown(label="Search Results")

        search_button.click(fn=search_questions, inputs=[search_input], outputs=[search_output])

    with gr.Tab("❌ Review Incorrect/Unknown"):
        with gr.Row():
            filter_radio = gr.Radio(
                label="Filter",
                choices=["All", "Incorrect", "Unknown"],
                value="All",
                interactive=True,
            )
            refresh_filtered_btn = gr.Button("🔄 Refresh List", variant="primary")

        filtered_list_output = gr.Markdown(label="Filtered Buckets")

        refresh_filtered_btn.click(fn=list_incorrect_or_unknown, inputs=[filter_radio], outputs=[filtered_list_output])

        with gr.Row():
            selected_bucket_input = gr.Number(label="Bucket Number", value=0, minimum=0, step=1)
            view_selected_btn = gr.Button("🔍 View Selected Bucket", variant="secondary")

        # Outputs for viewing the selected bucket (reuse existing viewer)
        selected_status_output = gr.Textbox(label="Status", interactive=False, max_lines=1)
        selected_summary_output = gr.Markdown(label="Bucket Summary")
        selected_prompts_output = gr.Markdown(label="All Prompts")

        view_selected_btn.click(
            fn=view_bucket,
            inputs=[selected_bucket_input],
            outputs=[selected_summary_output, selected_prompts_output, selected_status_output],
        )

    with gr.Tab("ℹ️ Info"):
        gr.Markdown(f"""
        ## About This Enhanced Tool
        
        This tool helps you navigate through prompt logs from hypothesis validation experiments with additional features:
        
        **New Features:**
        - **📁 Directory Integration**: Automatically scans `finance_bench_logs/` and `results/` directories
        - **🔽 Dropdown Selection**: Easy file selection with dropdown menus
        - **✅❌ Correctness Tagging**: See which buckets are correct/incorrect based on results
        - **🏆 Gold Answers**: View the expected correct answers alongside predictions
        - **📊 Statistics**: Get overall accuracy and performance metrics
        - **🔄 Refresh**: Update file lists to see new files
        
        **How to Use:**
        1. **Load Files**: Select files from the dropdowns in the "Load Files" tab
           - Prompt logs from: `/finance_bench_logs/`
           - Results from: `/results/`
        2. **View Statistics**: Check overall performance metrics
        3. **Browse Buckets**: Navigate through individual question workflows
        4. **Search**: Find specific questions or topics
        
        **Available Directories:**
        - **Prompt Logs**: `{len(get_prompt_log_files())}` files in `finance_bench_logs/`
        - **Results**: `{len(get_results_files())}` files in `results/`
        
        **Supported File Formats:**
        - Prompt logs: JSONL format with template_name fields
        - Results: JSON format with correctness indicators and gold answers
        
        **Bucket Organization:**
        Each bucket represents a complete workflow for one question, containing:
        1. Schema generation prompts
        2. Data extraction prompts  
        3. SQL merging prompts
        4. Final answer generation prompts
        
        **File Matching:**
        The tool automatically tries to match buckets with results using:
        - FinanceBench IDs
        - Question text similarity
        - Positional matching (bucket order = result order)
        """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7863, share=False, debug=True)
