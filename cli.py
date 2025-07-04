import argparse
from agent.agent import Agent
from computers.config import *
from computers.default import *
from memory_providers import FileMemoryProvider
from computers import computers_config


def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response.lower().strip() == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Select a computer environment from the available options."
    )
    parser.add_argument(
        "--computer",
        choices=computers_config.keys(),
        help="Choose the computer environment to use.",
        default="local-playwright",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Initial input to use instead of asking the user.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images during the execution.",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        help="Start the browsing session with a specific URL (only for browser environments).",
        default="https://bing.com",
    )
    parser.add_argument(
        "--memory-file",
        type=str,
        help="Path to the memory file for the file memory provider.",
        default=None,
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice input and output (requires microphone and speakers).",
    )
    args = parser.parse_args()
    ComputerClass = computers_config[args.computer]
    
    # initialize memory providers
    memory_providers = []
    if args.memory_file:
        memory_providers.append(FileMemoryProvider(args.memory_file))

    # VoiceIO is imported lazily to avoid requiring additional dependencies when --voice is not used
    if args.voice:
        from voice_io import VoiceIO  # noqa: E402

    with ComputerClass() as computer:
        # set up step handler (prints always, optionally speaks)
        if args.voice:
            voice_io = VoiceIO()

            def _step_handler(msg: str):
                print(msg)
                try:
                    voice_io.speak(msg)
                except Exception as e:
                    print(f"[VoiceIO] Failed to speak step: {e}")

            step_handler = _step_handler
        else:
            step_handler = print

        agent = Agent(
            computer=computer,
            acknowledge_safety_check_callback=acknowledge_safety_check_callback,
            memory_providers=memory_providers,
            step_handler=step_handler,
        )

        items: list[dict] = []

        # open browser at start url if applicable
        if args.computer in ["browserbase", "local-playwright"]:
            if not args.start_url.startswith("http"):
                args.start_url = "https://" + args.start_url
            agent.computer.goto(args.start_url)

        while True:
            try:
                if args.voice:
                    # record voice and transcribe
                    voice_io.play_beep()
                    step_handler("Press Enter to start recording…")
                    input()  # wait for enter
                    print("Recording... Speak now.")
                    voice_io.play_beep()
                    wav_path = voice_io.record_audio(duration=5)  # record 5 seconds
                    voice_io.play_beep()
                    user_input = voice_io.speech_to_text(wav_path)
                    print(f"You said: {user_input}")
                else:
                    user_input = args.input or input("> ")
                if user_input.strip().lower() == "exit":
                    break
            except EOFError as e:
                print(f"An error occurred: {e}")
                break

            items.append({"role": "user", "content": user_input})

            # run with full history; Agent will inject memory providers automatically
            output_items = agent.run_full_turn(
                items,
                print_steps=True,
                show_images=args.show,
                debug=args.debug,
            )

            items += output_items

            # reset --input after first loop
            args.input = None


if __name__ == "__main__":
    main()
