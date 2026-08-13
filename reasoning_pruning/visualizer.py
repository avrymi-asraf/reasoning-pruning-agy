"""Visualization tools for Reasoning Pruning.

Provides rich terminal rendering, notebook-native HTML formatting, and an interactive
Gradio app for inspecting reasoning traces, pruning decisions, and model battles.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Union

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reasoning_pruning.types import EvalResult, EvalSample, PruneDecision, ReasoningTrace

console = Console()


def render_trace_diff(
    trace: ReasoningTrace,
    decision: PruneDecision,
    as_html: bool = False,
) -> Union[str, Panel]:
    """Visualize a reasoning trace with pruning decisions highlighted.

    What it does:
        Highlights the reasoning steps:
        - Useful Prefix: Green background / text
        - Skipped Step(s): Red background / strikethrough (with reason tooltip/box)
        - Target Continuation: Bright Cyan / Blue (the next useful step)
        - Untouched Steps: Neutral text

    When to reach for it:
        - In a Jupyter/Colab notebook cell: `display(HTML(render_trace_diff(trace, decision, as_html=True)))`
        - In terminal or CLI: `rich.print(render_trace_diff(trace, decision))`

    Parameters:
        trace: The ReasoningTrace instance.
        decision: The corresponding PruneDecision.
        as_html: If True, returns a self-contained HTML snippet suitable for IPython display.
            If False, returns a Rich Panel renderable for terminal output.

    Returns:
        An HTML string (if as_html=True) or a Rich Panel object.
    """
    if as_html:
        return _render_trace_diff_html(trace, decision)
    return _render_trace_diff_rich(trace, decision)


def _render_trace_diff_rich(trace: ReasoningTrace, decision: PruneDecision) -> Panel:
    """Build a Rich terminal panel for the trace diff."""
    content = Text()

    content.append(f"Question: {trace.question}\n\n", style="bold yellow")

    if trace.prefix:
        content.append(f"[Prefix]: {trace.prefix}\n\n", style="dim green")

    content.append("Reasoning Steps:\n", style="bold underline")

    for i, step in enumerate(trace.steps):
        is_skipped = decision.can_skip and (decision.skip_start_idx <= i <= decision.skip_end_idx)
        is_target = decision.can_skip and (i == decision.skip_end_idx + 1)
        is_useful_prefix = decision.can_skip and (i < decision.skip_start_idx)

        step_header = f"[{i}] "
        if is_skipped:
            content.append(step_header, style="bold red")
            content.append(f"{step}  ◀ [PRUNED / SKIPPED]\n", style="strike red")
        elif is_target:
            content.append(step_header, style="bold cyan")
            content.append(f"{step}  ◀ [TARGET TRANSITION]\n", style="bold cyan")
        elif is_useful_prefix:
            content.append(step_header, style="green")
            content.append(f"{step}\n", style="green")
        else:
            content.append(step_header, style="dim white")
            content.append(f"{step}\n", style="white")

    if decision.can_skip:
        content.append(f"\nSkip Reason ({decision.decision_model}): ", style="bold magenta")
        content.append(f"{decision.reason}\n", style="italic")
    else:
        content.append("\nPruning Result: ", style="bold")
        content.append(f"No skippable steps found. ({decision.reason})\n", style="dim")

    return Panel(content, title="[bold blue]Reasoning Pruning Trace[/bold blue]", border_style="blue")


def _render_trace_diff_html(trace: ReasoningTrace, decision: PruneDecision) -> str:
    """Generate modern, self-contained HTML for notebook rendering."""
    q_safe = html.escape(trace.question)
    steps_html = []

    for i, step in enumerate(trace.steps):
        step_safe = html.escape(step)
        is_skipped = decision.can_skip and (decision.skip_start_idx <= i <= decision.skip_end_idx)
        is_target = decision.can_skip and (i == decision.skip_end_idx + 1)
        is_useful_prefix = decision.can_skip and (i < decision.skip_start_idx)

        if is_skipped:
            badge = '<span style="background:#ef4444;color:#fff;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">PRUNED</span>'
            style = "background: rgba(239, 68, 68, 0.12); border-left: 4px solid #ef4444; text-decoration: line-through; color: #b91c1c;"
            steps_html.append(f'<div style="padding:6px 12px; margin:4px 0; border-radius:4px; {style}"><strong>[{i}]</strong> {step_safe} {badge}</div>')
        elif is_target:
            badge = '<span style="background:#0284c7;color:#fff;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">NEXT USEFUL STEP</span>'
            style = "background: rgba(2, 132, 199, 0.12); border-left: 4px solid #0284c7; color: #0369a1; font-weight: 600;"
            steps_html.append(f'<div style="padding:6px 12px; margin:4px 0; border-radius:4px; {style}"><strong>[{i}]</strong> {step_safe} {badge}</div>')
        elif is_useful_prefix:
            style = "background: rgba(34, 197, 94, 0.08); border-left: 4px solid #22c55e; color: #15803d;"
            steps_html.append(f'<div style="padding:6px 12px; margin:4px 0; border-radius:4px; {style}"><strong>[{i}]</strong> {step_safe}</div>')
        else:
            style = "background: #f8fafc; border-left: 4px solid #cbd5e1; color: #334155;"
            steps_html.append(f'<div style="padding:6px 12px; margin:4px 0; border-radius:4px; {style}"><strong>[{i}]</strong> {step_safe}</div>')

    reason_box = ""
    if decision.can_skip:
        reason_box = f"""
        <div style="margin-top:12px; padding:10px 14px; background:#faf5ff; border:1px solid #e9d5ff; border-radius:6px; font-size:13px; color:#6b21a8;">
            <strong>Pruning Justification ({html.escape(decision.decision_model)}):</strong> {html.escape(decision.reason)}
        </div>
        """
    else:
        reason_box = f"""
        <div style="margin-top:12px; padding:10px 14px; background:#f1f5f9; border-radius:6px; font-size:13px; color:#475569;">
            <em>No skippable steps detected. ({html.escape(decision.reason)})</em>
        </div>
        """

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width:800px; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 1px 3px rgba(0,0,0,0.05); background:#fff;">
        <div style="font-size:14px; font-weight:700; color:#1e293b; margin-bottom:8px;">
            <span style="color:#2563eb;">Question:</span> {q_safe}
        </div>
        <div style="margin-top:10px;">
            {''.join(steps_html)}
        </div>
        {reason_box}
    </div>
    """


def render_comparison(
    sample: Union[EvalSample, Dict[str, Any]],
    as_html: bool = False,
) -> Union[str, Panel]:
    """Render side-by-side comparison between Base model and Pruned model.

    What it does:
        Visualizes generation lengths, token savings, reasoning steps, and accuracy.

    Parameters:
        sample: An EvalSample or dictionary with question, base_response, pruned_response, etc.
        as_html: If True, formats as HTML for notebooks.
    """
    if isinstance(sample, dict):
        q = sample.get("question", "")
        b_resp = sample.get("base_response", "")
        p_resp = sample.get("pruned_response", "")
        b_tok = sample.get("base_tokens", len(b_resp.split()))
        p_tok = sample.get("pruned_tokens", len(p_resp.split()))
        savings = sample.get("token_savings_pct", (1 - p_tok / max(b_tok, 1)) * 100)
        gt = sample.get("ground_truth")
    else:
        q = sample.question
        b_resp = sample.base_response
        p_resp = sample.pruned_response
        b_tok = sample.base_tokens
        p_tok = sample.pruned_tokens
        savings = sample.token_savings_pct
        gt = sample.ground_truth

    if as_html:
        gt_html = f"<div style='margin-bottom:8px;font-size:12px;color:#64748b;'><strong>Ground Truth:</strong> {html.escape(str(gt))}</div>" if gt else ""
        return f"""
        <div style="font-family: sans-serif; border:1px solid #e2e8f0; border-radius:8px; padding:16px; background:#fff; margin:10px 0;">
            <div style="font-weight:bold; margin-bottom:6px; color:#0f172a;">Q: {html.escape(q)}</div>
            {gt_html}
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:12px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px;">
                    <div style="font-weight:bold; color:#475569; margin-bottom:6px;">Base Model ({b_tok} tokens)</div>
                    <pre style="white-space:pre-wrap; font-size:12px; margin:0; color:#334155;">{html.escape(b_resp)}</pre>
                </div>
                <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:12px;">
                    <div style="font-weight:bold; color:#166534; margin-bottom:6px;">Pruned Model ({p_tok} tokens, <span style="color:#15803d;">-{savings:.1f}%</span>)</div>
                    <pre style="white-space:pre-wrap; font-size:12px; margin:0; color:#14532d;">{html.escape(p_resp)}</pre>
                </div>
            </div>
        </div>
        """

    table = Table(title="Model Generation Comparison", show_header=True)
    table.add_column("Metric / Trace", style="cyan")
    table.add_column(f"Base Model ({b_tok} tok)", style="white")
    table.add_column(f"Pruned Model ({p_tok} tok, -{savings:.1f}%)", style="green")

    table.add_row("Reasoning Output", b_resp[:300] + ("..." if len(b_resp) > 300 else ""), p_resp[:300] + ("..." if len(p_resp) > 300 else ""))
    return Panel(table, title=f"[bold]Question: {q[:60]}...[/bold]")


def launch_viewer(
    dataset: Optional[Any] = None,
    eval_result: Optional[EvalResult] = None,
    share: bool = False,
    port: int = 7860,
) -> Any:
    """Launch an interactive Gradio dashboard to visually explore pruning traces and evaluations.

    What it does:
        Starts a web interface with:
        1. Live Pruning Playground: Test pruning prompts on any question in real time.
        2. Dataset Explorer: Inspect generated PT dataset rows with colored diffs.
        3. Model Battle Viewer: Side-by-side generation and token savings comparison.

    When to reach for it:
        - When running in Google Colab (`share=True` generates a public link).
        - For interactive team demos and deep qualitative inspection.

    Parameters:
        dataset: Optional Hugging Face Dataset of TransitionExamples.
        eval_result: Optional EvalResult from `evaluate_models`.
        share: Whether to generate a public shareable URL (useful in Colab).
        port: Local port to serve the app.

    Returns:
        The running Gradio app instance.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for launch_viewer. Install with: uv add gradio")

    from reasoning_pruning.decision import find_first_skip
    from reasoning_pruning.generation import generate_trace, rollout_pruning

    def _interactive_prune(q: str, gen_model: str, dec_model: str, max_depth: int):
        rollout = rollout_pruning(
            question=q,
            generator_model=gen_model,
            decision_model=dec_model,
            max_depth=int(max_depth),
        )
        # Render HTML for all intermediate traces
        html_cards = []
        for i, (tr, dec) in enumerate(zip(rollout.traces, rollout.decisions)):
            html_cards.append(f"<h4 style='margin-top:16px;'>Depth {i+1} Rollout</h4>")
            html_cards.append(render_trace_diff(tr, dec, as_html=True))

        summary_text = (
            f"Original Steps: {rollout.original_step_count} | "
            f"Final Steps: {rollout.final_step_count} | "
            f"Compression: {rollout.compression_ratio*100:.1f}% | "
            f"Extracted PT Pairs: {len(rollout.transitions)}"
        )
        return "".join(html_cards), summary_text

    with gr.Blocks(title="Reasoning Pruning Inspector") as demo:
        gr.Markdown("# 🧠 Reasoning Pruning (RP) Visualizer")

        with gr.Tab("🔬 Live Pruning Playground"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_input = gr.Textbox(
                        label="Input Question / Math Problem",
                        value="Janet buys 3 packs of 12 eggs. She bakes 2 cakes using 4 eggs each. How many eggs does she have left?",
                        lines=3,
                    )
                    gen_dropdown = gr.Dropdown(
                        label="Generator Model (G)",
                        choices=["gpt-4o-mini", "claude-3-5-haiku-20241022", "deepseek/deepseek-chat"],
                        value="gpt-4o-mini",
                    )
                    dec_dropdown = gr.Dropdown(
                        label="Decision Model (D)",
                        choices=["gpt-4o-mini", "claude-3-5-haiku-20241022", "deepseek/deepseek-chat"],
                        value="gpt-4o-mini",
                    )
                    depth_slider = gr.Slider(1, 5, value=2, step=1, label="Max Pruning Depth")
                    run_btn = gr.Button("⚡ Run Pruning Rollout", variant="primary")
                    stats_box = gr.Textbox(label="Compression Summary", interactive=False)

                with gr.Column(scale=2):
                    diff_output = gr.HTML(label="Pruning Trace Diff")

            run_btn.click(
                _interactive_prune,
                inputs=[q_input, gen_dropdown, dec_dropdown, depth_slider],
                outputs=[diff_output, stats_box],
            )

        if dataset is not None:
            with gr.Tab("📁 Dataset Browser"):
                gr.Markdown(f"Browsing **{len(dataset)}** Pruning-Transition examples.")
                sample_idx = gr.Slider(0, max(0, len(dataset) - 1), value=0, step=1, label="Example Index")
                sample_display = gr.JSON(label="Sample JSON")

                def _show_sample(idx):
                    if idx < len(dataset):
                        return dict(dataset[int(idx)])
                    return {}

                sample_idx.change(_show_sample, inputs=[sample_idx], outputs=[sample_display])

    demo.launch(share=share, server_port=port, inbrowser=False)
    return demo
