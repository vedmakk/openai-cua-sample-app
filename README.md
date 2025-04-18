# Computer Using Agent Sample App

Get started building a [Computer Using Agent (CUA)](https://platform.openai.com/docs/guides/tools-computer-use) with the OpenAI API.

> [!CAUTION]  
> Computer use is in preview. Because the model is still in preview and may be susceptible to exploits and inadvertent mistakes, we discourage trusting it in authenticated environments or for high-stakes tasks.

## Set Up & Run

Set up python env and install dependencies.

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run CLI to let CUA use a local browser window, using [playwright](https://playwright.dev/). (Stop with CTRL+C)

```shell
python cli.py --computer local-playwright

python cli.py --computer local-playwright --start-url https://maildrop.cc --brain-file ./memory/brain.txt
```

> [!NOTE]  
> The first time you run this, if you haven't used Playwright before, you will be prompted to install dependencies. Execute the command suggested, which will depend on your OS.

Other included sample [computer environments](#computer-environments):

- [Docker](https://docker.com/) (containerized desktop)
- [Browserbase](https://www.browserbase.com/) (remote browser, requires account)
- [Scrapybara](https://scrapybara.com) (remote browser or computer, requires account)
- ...or implement your own `Computer`!

## Overview

The computer use tool and model are available via the [Responses API](https://platform.openai.com/docs/api-reference/responses). At a high level, CUA will look at a screenshot of the computer interface and recommend actions. Specifically, it sends `computer_call`(s) with `actions` like `click(x,y)` or `type(text)` that you have to execute on your environment, and then expects screenshots of the outcomes.

You can learn more about this tool in the [Computer use guide](https://platform.openai.com/docs/guides/tools-computer-use).

## Abstractions

This repository defines two lightweight abstractions to make interacting with CUA agents more ergonomic. Everything works without them, but they provide a convenient separation of concerns.

| Abstraction | File                    | Description                                                                                                                                                                                                  |
| ----------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Computer`  | `computers/computer.py` | Defines a `Computer` interface for various environments (local desktop, remote browser, etc.). An implementation of `Computer` is responsible for executing any `computer_action` sent by CUA (clicks, etc). |
| `Agent`     | `agent/agent.py`        | Simple, familiar agent loop – implements `run_full_turn()`, which just keeps calling the model until all computer actions and function calls are handled.                                                    |

## CLI Usage

The CLI (`cli.py`) is the easiest way to get started with CUA. It accepts the following arguments:

- `--computer`: The computer environment to use. See the [Computer Environments](#computer-environments) section below for options. By default, the CLI will use the `local-playwright` environment.
- `--input`: The initial input to the agent (optional: the CLI will prompt you for input if not provided)
- `--debug`: Enable debug mode.
- `--show`: Show images (screenshots) during the execution.
- `--start-url`: Start the browsing session with a specific URL (only for browser environments). By default, the CLI will start the browsing session with `https://bing.com`.
- `--brain-file`: Path to the brain/memory file for the agent. If provided, the agent will use this file to persist and retrieve memory across the session.

### Run examples (optional)

The `examples` folder contains more examples of how to use CUA.

```shell
python -m examples.weather_example
```

For reference, the file `simple_cua_loop.py` implements the basics of the CUA loop.

You can run it with:

```shell
python simple_cua_loop.py
```

## Computer Environments

CUA can work with any `Computer` environment that can handle the [CUA actions](https://platform.openai.com/docs/api-reference/responses/object#responses/object-output):

| Action                             | Example                         |
| ---------------------------------- | ------------------------------- |
| `click(x, y, button="left")`       | `click(24, 150)`                |
| `double_click(x, y)`               | `double_click(24, 150)`         |
| `scroll(x, y, scroll_x, scroll_y)` | `scroll(24, 150, 0, -100)`      |
| `type(text)`                       | `type("Hello, World!")`         |
| `wait(ms=1000)`                    | `wait(2000)`                    |
| `move(x, y)`                       | `move(24, 150)`                 |
| `keypress(keys)`                   | `keypress(["CTRL", "C"])`       |
| `drag(path)`                       | `drag([[24, 150], [100, 200]])` |

This sample app provides a set of implemented `Computer` examples, but feel free to add your own!

| Computer            | Option             | Type      | Description                       | Requirements                                                     |
| ------------------- | ------------------ | --------- | --------------------------------- | ---------------------------------------------------------------- |
| `LocalPlaywright`   | local-playwright   | `browser` | Local browser window              | [Playwright SDK](https://playwright.dev/)                        |
| `Docker`            | docker             | `linux`   | Docker container environment      | [Docker](https://docs.docker.com/engine/install/) running        |
| `Browserbase`       | browserbase        | `browser` | Remote browser environment        | [Browserbase](https://www.browserbase.com/) API key in `.env`    |
| `ScrapybaraBrowser` | scrapybara-browser | `browser` | Remote browser environment        | [Scrapybara](https://scrapybara.com/dashboard) API key in `.env` |
| `ScrapybaraUbuntu`  | scrapybara-ubuntu  | `linux`   | Remote Ubuntu desktop environment | [Scrapybara](https://scrapybara.com/dashboard) API key in `.env` |

Using the CLI, you can run the sample app with different computer environments using the options listed above:

```shell
python cli.py --show --computer <computer-option>
```

For example, to run the sample app with the `Docker` computer environment, you can run:

```shell
python cli.py --show --computer docker
```

### Docker Setup

If you want to run the sample app with the `Docker` computer environment, you need to build and run a local Docker container.

Open a new shell to build and run the Docker image. The first time you do this, it may take a few minutes, but subsequent runs should be much faster. Once the logs stop, proceed to the next setup step. To stop the container, press CTRL+C on the terminal where you ran the command below.

```shell
docker build -t cua-sample-app .
docker run --rm -it --name cua-sample-app -p 5900:5900 --dns=1.1.1.3 -e DISPLAY=:99 cua-sample-app
```

> [!NOTE]  
> We use `--dns=1.1.1.3` to restrict accessible websites to a smaller, safer set. We highly recommend you take similar safety precautions.

> [!WARNING]  
> If you get the below error, then you need to kill that container.
>
> ```
> docker: Error response from daemon: Conflict. The container name "/cua-sample-app" is already in use by container "e72fcb962b548e06a9dcdf6a99bc4b49642df2265440da7544330eb420b51d87"
> ```
>
> Kill that container and try again.
>
> ```shell
> docker rm -f cua-sample-app
> ```

### Hosted environment setup

This repository contains example implementations of third-party hosted environments.
To use these, you will need to set up an account with the service by following the links aboveand add your API key to the `.env` file.

## Function Calling

The `Agent` class accepts regular function schemas in `tools` – it will return a hard-coded value for any invocations.

However, if you pass in any `tools` that are also defined in your `Computer` methods, in addition to the required `Computer` methods, they will be routed to your `Computer` to be handled when called. **This is useful for cases where screenshots often don't capture the search bar or back arrow, so CUA may get stuck. So instead, you can provide a `back()` or `goto(url)` functions.** See `examples/playwright_with_custom_functions.py` for an example.

## Risks & Safety considerations

This repository provides example implementations with basic safety measures in place.

We recommend reviewing the best practices outlined in our [guide](https://platform.openai.com/docs/guides/tools-computer-use#risks-and-safety), and making sure you understand the risks involved with using this tool.
