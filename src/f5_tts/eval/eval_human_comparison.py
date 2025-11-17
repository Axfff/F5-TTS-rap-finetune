import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import gradio as gr


def load_audio_pairs(baseline_dir, finetuned_dir):
    """Load pairs of baseline and finetuned audio files"""
    baseline_path = Path(baseline_dir)
    finetuned_path = Path(finetuned_dir)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Finetuned directory not found: {finetuned_dir}")

    baseline_files = {f.stem: f for f in baseline_path.glob("*.wav")}
    finetuned_files = {f.stem: f for f in finetuned_path.glob("*.wav")}

    common_files = set(baseline_files.keys()) & set(finetuned_files.keys())

    if not common_files:
        raise ValueError("No matching audio files found between baseline and finetuned directories")

    audio_pairs = []
    for file_id in sorted(common_files):
        # Randomly assign which audio goes to position A or B
        swap = random.choice([True, False])
        if swap:
            audio_pairs.append({
                "id": file_id,
                "audio_a": str(finetuned_files[file_id]),
                "audio_b": str(baseline_files[file_id]),
                "a_is": "finetuned",
                "b_is": "baseline"
            })
        else:
            audio_pairs.append({
                "id": file_id,
                "audio_a": str(baseline_files[file_id]),
                "audio_b": str(finetuned_files[file_id]),
                "a_is": "baseline",
                "b_is": "finetuned"
            })

    return audio_pairs


def create_evaluation_ui(baseline_dir, finetuned_dir, output_csv):
    """Create Gradio UI for human evaluation"""

    audio_pairs = load_audio_pairs(baseline_dir, finetuned_dir)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'audio_id', 'choice', 'a_is', 'b_is', 'notes'])

    def get_audio_pair(index):
        """Get audio pair at given index"""
        if index >= len(audio_pairs):
            return None, None, f"Evaluation complete! ({len(audio_pairs)}/{len(audio_pairs)})", "", "", index

        pair = audio_pairs[index]
        progress = f"Sample {index + 1}/{len(audio_pairs)}: {pair['id']}"
        return pair['audio_a'], pair['audio_b'], progress, pair['id'], "", index

    def save_choice(index, choice, notes):
        """Save user's choice to CSV"""
        if index >= len(audio_pairs):
            return index, "Evaluation already complete!", ""

        pair = audio_pairs[index]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Map user choice to actual model
        if choice == "a":
            actual_choice = pair['a_is']
        elif choice == "b":
            actual_choice = pair['b_is']
        else:  # equal
            actual_choice = "equal"

        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, pair['id'], choice, pair['a_is'], pair['b_is'], notes])

        new_index = index + 1

        # Reveal which was which
        reveal_msg = f"You chose: {choice.upper() if choice != 'equal' else 'Equal'}\n"
        reveal_msg += f"Audio A was: {pair['a_is'].upper()}\n"
        reveal_msg += f"Audio B was: {pair['b_is'].upper()}\n"
        reveal_msg += f"Actual choice: {actual_choice.upper()}"

        if new_index >= len(audio_pairs):
            return new_index, f"{reveal_msg}\n\nEvaluation complete! All {len(audio_pairs)} samples rated.", reveal_msg

        return new_index, f"{reveal_msg}\n\nSaved! Moving to next sample...", reveal_msg

    def on_choice(index, choice, notes):
        """Handle choice button click"""
        new_index, message, reveal = save_choice(index, choice, notes)
        audio_a, audio_b, progress, audio_id, _, _ = get_audio_pair(new_index)
        return new_index, audio_a, audio_b, progress, audio_id, "", message, reveal

    with gr.Blocks(title="Rap Model Evaluation") as app:
        current_index = gr.State(0)
        gr.Markdown("""
        # Rap Model Human Evaluation

        Listen to both audio samples (Audio A and Audio B) and choose which one sounds like a better rapper.

        **Note**: The baseline and finetuned models are randomly assigned to Audio A or Audio B.
        After you make your choice, the system will reveal which was which.

        Click the button corresponding to the better performance, or "Equal" if they're the same.
        """)

        with gr.Row():
            progress_text = gr.Textbox(label="Progress", interactive=False, value="")
            audio_id_text = gr.Textbox(label="Audio ID", interactive=False, value="")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Audio A")
                audio_a = gr.Audio(label="Audio A", type="filepath")

            with gr.Column():
                gr.Markdown("### Audio B")
                audio_b = gr.Audio(label="Audio B", type="filepath")

        notes_input = gr.Textbox(
            label="Notes (optional)",
            placeholder="Add any observations about the samples...",
            lines=2
        )

        with gr.Row():
            a_btn = gr.Button("Audio A is Better", variant="primary", size="lg")
            equal_btn = gr.Button("Equal / No Preference", variant="secondary", size="lg")
            b_btn = gr.Button("Audio B is Better", variant="primary", size="lg")

        status_text = gr.Textbox(label="Status", interactive=False)
        reveal_text = gr.Textbox(label="Reveal", interactive=False, value="")

        app.load(
            get_audio_pair,
            inputs=[current_index],
            outputs=[audio_a, audio_b, progress_text, audio_id_text, reveal_text, current_index]
        )

        a_btn.click(
            lambda idx, notes: on_choice(idx, "a", notes),
            inputs=[current_index, notes_input],
            outputs=[current_index, audio_a, audio_b, progress_text, audio_id_text, notes_input, status_text, reveal_text]
        )

        b_btn.click(
            lambda idx, notes: on_choice(idx, "b", notes),
            inputs=[current_index, notes_input],
            outputs=[current_index, audio_a, audio_b, progress_text, audio_id_text, notes_input, status_text, reveal_text]
        )

        equal_btn.click(
            lambda idx, notes: on_choice(idx, "equal", notes),
            inputs=[current_index, notes_input],
            outputs=[current_index, audio_a, audio_b, progress_text, audio_id_text, notes_input, status_text, reveal_text]
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Human evaluation UI for comparing baseline and finetuned models")

    parser.add_argument("--baseline_dir", type=str, required=True, help="Directory containing baseline model outputs")
    parser.add_argument("--finetuned_dir", type=str, required=True, help="Directory containing finetuned model outputs")
    parser.add_argument("--output_csv", type=str, default="results/human_evaluation.csv", help="Path to save evaluation results")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI")
    parser.add_argument("--share", action="store_true", help="Create a public share link")

    args = parser.parse_args()

    print(f"\nLoading audio pairs from:")
    print(f"  Baseline:  {args.baseline_dir}")
    print(f"  Finetuned: {args.finetuned_dir}")
    print(f"\nResults will be saved to: {args.output_csv}")

    app = create_evaluation_ui(args.baseline_dir, args.finetuned_dir, args.output_csv)

    print(f"\nLaunching web UI on port {args.port}...")
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()


"""example usage

python src/f5_tts/eval/eval_human_comparison.py \
    --baseline_dir results/rap_evaluation/baseline \
    --finetuned_dir results/rap_evaluation/finetuned \
    --output_csv results/rap_evaluation/human_eval.csv \
    --port 7860

"""
