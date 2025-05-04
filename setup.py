import os
import sys
from typing import Optional

import typer
import torch
from rich.console import Console
from rich.panel import Panel

from config import get_default_config, load_config
from inference.predictor import HallucinationPredictor
from models.classifier import create_hallucination_classifier
from models.encoder import load_encoder_model

app = typer.Typer(
    name="hallucination-hunter",
    help="Detect hallucinations in chain-of-thought reasoning",
    add_completion=False,
)

console = Console()


@app.command("predict")
def predict(
    question: str = typer.Argument(..., help="The mathematical question"),
    reasoning: str = typer.Argument(..., help="Chain-of-thought reasoning"),
    model_path: str = typer.Option(
        "models/hallucination_classifier",
        "--model-path",
        "-m",
        help="Path to the model checkpoint",
    ),
    model_name: str = typer.Option(
        "roberta-base",
        "--model-name",
        "-n",
        help="Name of the transformer model",
    ),
    confidence_threshold: float = typer.Option(
        0.7,
        "--confidence-threshold",
        "-c",
        help="Confidence threshold for reliable predictions",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to run inference on (cpu, cuda, cuda:0, etc.)",
    ),
):
    console.print(
        Panel.fit(
            "Hallucination Hunter - Analyzing Chain-of-Thought Reasoning",
            style="bold blue",
        )
    )

    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Reasoning:[/bold]\n{reasoning}")
    console.print("\n[bold]Analyzing reasoning...[/bold]")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    try:
        encoder, tokenizer = load_encoder_model(model_name)
        model = create_hallucination_classifier(model_name)

        if not os.path.exists(model_path):
            console.print(f"[bold red]Error:[/bold red] Model not found at {model_path}")
            sys.exit(1)

        model.load_state_dict(torch.load(model_path, map_location=device))

        predictor = HallucinationPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        predictions = predictor.predict_reasoning_chain(
            question=question,
            reasoning=reasoning,
        )

        for i, pred in enumerate(predictions):
            step = pred["step"]
            is_hallucination = pred["is_hallucination"]
            confidence = pred["confidence"]
            is_reliable = pred["is_reliable"]

            if is_hallucination:
                if is_reliable:
                    status = "[bold red]HALLUCINATION[/bold red]"
                else:
                    status = "[bold orange3]POTENTIAL HALLUCINATION[/bold orange3]"
            else:
                if is_reliable:
                    status = "[bold green]GROUNDED[/bold green]"
                else:
                    status = "[bold blue]LIKELY GROUNDED[/bold blue]"

            console.print(f"\n[bold]Step {i+1}:[/bold] {status} ({confidence:.2f})")
            console.print(f"  {step}")

        num_steps = len(predictions)
        num_hallucinations = sum(1 for p in predictions if p["is_hallucination"])
        hallucination_rate = (num_hallucinations / num_steps) * 100 if num_steps > 0 else 0

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Total steps: {num_steps}")
        console.print(f"  Hallucinated steps: {num_hallucinations}")
        console.print(f"  Hallucination rate: {hallucination_rate:.1f}%")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@app.command("serve")
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the API server",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind the API server",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of worker processes",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload",
    ),
    ui_port: int = typer.Option(
        8501,
        "--ui-port",
        "-u",
        help="Port to bind the Streamlit UI",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    console.print(
        Panel.fit(
            "Hallucination Hunter - Starting API and UI Services",
            style="bold blue",
        )
    )

    if config_path is not None:
        try:
            config = load_config(config_path)
            console.print(f"[bold green]Loaded configuration from {config_path}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error loading config:[/bold red] {str(e)}")
            console.print("[yellow]Using default configuration[/yellow]")
            config = get_default_config()
    else:
        config = get_default_config()

    import subprocess
    import time
    import signal
    import sys

    processes = []

    def cleanup(sig, frame):
        console.print("[bold yellow]Shutting down services...[/bold yellow]")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        api_cmd = [
            "uvicorn",
            "hallucination_hunter.api.main:app",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers),
            "--log-level", "info",
        ]

        if reload:
            api_cmd.append("--reload")

        console.print(f"[bold]Starting API server on {host}:{port}[/bold]")
        api_process = subprocess.Popen(api_cmd)
        processes.append(api_process)

        ui_cmd = [
            "streamlit",
            "run",
            "hallucination_hunter/ui/app.py",
            "--server.port", str(ui_port),
            "--server.address", host,
        ]

        console.print(f"[bold]Starting UI server on {host}:{ui_port}[/bold]")
        ui_process = subprocess.Popen(ui_cmd)
        processes.append(ui_process)

        console.print(
            f"\n[bold green]Services started successfully![/bold green]\n"
            f"  API: http://{host}:{port}\n"
            f"  UI: http://{host}:{ui_port}\n"
            f"  API Documentation: http://{host}:{port}/docs\n"
            f"\nPress Ctrl+C to stop the services."
        )

        # Wait for processes to finish or be terminated
        for p in processes:
            p.wait()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        cleanup(None, None)


if __name__ == "__main__":
    app()
