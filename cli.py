import argparse
from agent.agent import Agent
from memory_providers import FileMemoryProvider
from computers import (
    BrowserbaseBrowser,
    ScrapybaraBrowser,
    ScrapybaraUbuntu,
    LocalPlaywrightComputer,
    DockerComputer,
)

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
        choices=[
            "local-playwright",
            "docker",
            "browserbase",
            "scrapybara-browser",
            "scrapybara-ubuntu",
        ],
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
    args = parser.parse_args()

    computer_mapping = {
        "local-playwright": LocalPlaywrightComputer,
        "docker": DockerComputer,
        "browserbase": BrowserbaseBrowser,
        "scrapybara-browser": ScrapybaraBrowser,
        "scrapybara-ubuntu": ScrapybaraUbuntu,
    }

    ComputerClass = computer_mapping[args.computer]
    # initialize memory providers
    memory_providers = []
    if args.memory_file:
        memory_providers.append(FileMemoryProvider(args.memory_file))

    with ComputerClass() as computer:
        agent = Agent(
            computer=computer,
            acknowledge_safety_check_callback=acknowledge_safety_check_callback,
            memory_providers=memory_providers,
        )
        items = []


        if args.computer in ["browserbase", "local-playwright"]:
            if not args.start_url.startswith("http"):
                args.start_url = "https://" + args.start_url
            agent.computer.goto(args.start_url)
        while True:
            try:
                user_input = args.input or input("> ")
                if user_input == 'exit':
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
            args.input = None


if __name__ == "__main__":
    main()
