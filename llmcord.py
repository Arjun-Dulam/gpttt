import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
import time
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_30_DAYS = 30 * SECONDS_PER_DAY


def load_env_file(filename: str = ".env") -> None:
    if not os.path.exists(filename):
        return

    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_env_values(val) for key, val in value.items()}
    if isinstance(value, list):
        return [resolve_env_values(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    load_env_file()

    with open(filename, encoding="utf-8") as file:
        return resolve_env_values(yaml.safe_load(file))


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class MonitorState:
    start_time: datetime = field(default_factory=lambda: datetime.now().astimezone())
    messages_seen: int = 0
    llm_requests: int = 0
    failed_llm_requests: int = 0
    generated_messages: int = 0
    active_generations: int = 0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None


monitor_state = MonitorState()


def get_admin_ids(curr_config: dict[str, Any]) -> set[int]:
    return set(curr_config["permissions"]["users"]["admin_ids"])


def user_is_admin(user_id: int, curr_config: dict[str, Any]) -> bool:
    return user_id in get_admin_ids(curr_config)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, SECONDS_PER_DAY)
    hours, seconds = divmod(seconds, SECONDS_PER_HOUR)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts[:2])


def format_timestamp(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z") if dt else "Never"


def format_money(amount: float) -> str:
    return f"${amount:.4f}" if amount < 1 else f"${amount:.2f}"


def count_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(count_content_chars(item.get("text", "")) for item in content if isinstance(item, dict))
    return 0


def estimate_tokens(chars: int, multiplier: float) -> int:
    return max(0, round(chars * multiplier))


def get_model_prices(curr_config: dict[str, Any], model_name: str) -> tuple[float, float]:
    prices = (curr_config.get("cost_estimation") or {}).get("model_prices") or {}
    model_prices = prices.get(model_name) or {}
    return float(model_prices.get("input_per_million", 0)), float(model_prices.get("output_per_million", 0))


def calculate_cost(curr_config: dict[str, Any], model_name: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = get_model_prices(curr_config, model_name)
    return (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)


def estimate_usage_cost(
    curr_config: dict[str, Any],
    model_name: str,
    prompt_chars: int,
    response_chars: int,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> tuple[int, int, float]:
    cost_config = curr_config.get("cost_estimation") or {}
    input_token_count = input_tokens if input_tokens is not None else estimate_tokens(prompt_chars, float(cost_config.get("input_tokens_per_char", 0.25)))
    output_token_count = output_tokens if output_tokens is not None else estimate_tokens(response_chars, float(cost_config.get("output_tokens_per_char", 0.25)))
    return input_token_count, output_token_count, calculate_cost(curr_config, model_name, input_token_count, output_token_count)


def get_usage_log_path(curr_config: dict[str, Any]) -> str:
    return curr_config.get("usage_log_path", "usage.jsonl")


def append_usage_event(curr_config: dict[str, Any], event: dict[str, Any]) -> None:
    with open(get_usage_log_path(curr_config), "a", encoding="utf-8") as file:
        file.write(json.dumps(event, separators=(",", ":")) + "\n")


def load_usage_events(curr_config: dict[str, Any], since_seconds: int = SECONDS_PER_30_DAYS) -> list[dict[str, Any]]:
    path = get_usage_log_path(curr_config)
    if not os.path.exists(path):
        return []

    cutoff = time.time() - since_seconds
    events = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if float(event.get("timestamp", 0)) >= cutoff:
                events.append(event)

    return events


def summarize_costs(events: list[dict[str, Any]]) -> tuple[float, float, float]:
    now = time.time()

    def total_since(seconds: int) -> float:
        cutoff = now - seconds
        return sum(float(event.get("estimated_cost", 0)) for event in events if float(event.get("timestamp", 0)) >= cutoff)

    return total_since(SECONDS_PER_HOUR), total_since(SECONDS_PER_DAY), total_since(SECONDS_PER_30_DAYS)


def summarize_top_messengers(events: list[dict[str, Any]], limit: int = 5) -> list[tuple[int, int, float]]:
    users = {}

    for event in events:
        if not event.get("success"):
            continue

        user_id = int(event["user_id"])
        count, cost = users.get(user_id, (0, 0.0))
        users[user_id] = (count + 1, cost + float(event.get("estimated_cost", 0)))

    return sorted(((user_id, count, cost) for user_id, (count, cost) in users.items()), key=lambda item: (-item[1], -item[2], item[0]))[:limit]


def extract_usage_tokens(usage: Any) -> tuple[Optional[int], Optional[int]]:
    if usage is None:
        return None, None

    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

    return input_tokens, output_tokens


def build_monitor_embed(curr_config: dict[str, Any]) -> discord.Embed:
    events = load_usage_events(curr_config)
    hour_cost, day_cost, month_cost = summarize_costs(events)
    top_messengers = summarize_top_messengers(events)

    concurrency_limit = curr_config.get("global_concurrency_limit", 0)
    busy_threshold = concurrency_limit * 0.75 if concurrency_limit else float("inf")

    if monitor_state.last_error:
        health_label = "Error"
        color = discord.Color.red()
    elif monitor_state.active_generations >= busy_threshold:
        health_label = "Busy"
        color = discord.Color.gold()
    else:
        health_label = "Healthy"
        color = discord.Color.green()

    uptime = format_duration((datetime.now().astimezone() - monitor_state.start_time).total_seconds())
    latency_ms = round(discord_bot.latency * 1000)

    embed = discord.Embed(
        title="Bot Monitor",
        description=f"Status: **{health_label}**",
        color=color,
        timestamp=datetime.now().astimezone(),
    )

    embed.add_field(name="Model", value=f"`{curr_model}`", inline=False)
    embed.add_field(name="Uptime", value=uptime, inline=True)
    embed.add_field(name="Latency", value=f"{latency_ms} ms", inline=True)
    embed.add_field(name="Active", value=str(monitor_state.active_generations), inline=True)
    embed.add_field(name="Cache", value=f"{len(msg_nodes)} / {MAX_MESSAGE_NODES}", inline=True)
    embed.add_field(name="Messages Seen", value=f"{monitor_state.messages_seen:,}", inline=True)
    embed.add_field(name="LLM Requests", value=f"{monitor_state.llm_requests:,}", inline=True)
    embed.add_field(name="Failures", value=f"{monitor_state.failed_llm_requests:,}", inline=True)
    embed.add_field(name="Generated", value=f"{monitor_state.generated_messages:,}", inline=True)
    embed.add_field(name="Last Request", value=format_timestamp(monitor_state.last_request_time), inline=False)
    embed.add_field(
        name="Estimated Cost",
        value=f"Last hour: **{format_money(hour_cost)}**\nLast day: **{format_money(day_cost)}**\nLast 30 days: **{format_money(month_cost)}**",
        inline=True,
    )

    leaderboard = "\n".join(f"{index}. <@{user_id}> - {count} requests, {format_money(cost)}" for index, (user_id, count, cost) in enumerate(top_messengers, start=1))
    embed.add_field(name="Top Messengers, 30d", value=leaderboard or "No usage yet", inline=True)

    if monitor_state.last_error:
        embed.add_field(name="Last Error", value=monitor_state.last_error[:1024], inline=False)

    embed.set_footer(text="Cost is estimated from configured pricing")
    return embed


@discord_bot.tree.command(name="monitor", description="View bot health, usage, cost, and top messengers")
async def monitor_command(interaction: discord.Interaction) -> None:
    curr_config = await asyncio.to_thread(get_config)

    if not user_is_admin(interaction.user.id, curr_config):
        await interaction.response.send_message("You don't have permission to view bot monitoring.", ephemeral=True)
        return

    await interaction.response.send_message(embed=build_monitor_embed(curr_config), ephemeral=True)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in get_admin_ids(config):
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    monitor_state.messages_seen += 1

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in get_admin_ids(config)

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                if curr_node.role == "user" and (curr_node.text or curr_node.images):
                    curr_node.text = f"<@{curr_msg.author.id}>: {curr_node.text}"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                messages.append(dict(content=content, role=curr_node.role))

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    prompt_chars = sum(count_content_chars(message["content"]) for message in messages)
    input_tokens = output_tokens = None
    request_succeeded = False
    error_summary = None

    openai_kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=True,
        stream_options=dict(include_usage=True),
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
    )

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    monitor_state.llm_requests += 1
    monitor_state.active_generations += 1
    monitor_state.last_request_time = datetime.now().astimezone()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                chunk_input_tokens, chunk_output_tokens = extract_usage_tokens(getattr(chunk, "usage", None))
                input_tokens = chunk_input_tokens if chunk_input_tokens is not None else input_tokens
                output_tokens = chunk_output_tokens if chunk_output_tokens is not None else output_tokens

                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

        request_succeeded = True
        monitor_state.last_error = None

    except Exception as exc:
        logging.exception("Error while generating response")
        monitor_state.failed_llm_requests += 1
        error_summary = f"{type(exc).__name__}: {exc}"
        monitor_state.last_error = error_summary

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    monitor_state.active_generations = max(0, monitor_state.active_generations - 1)
    monitor_state.generated_messages += len(response_msgs)

    response_text = "".join(response_contents)
    input_tokens, output_tokens, estimated_cost = estimate_usage_cost(config, provider_slash_model, prompt_chars, len(response_text), input_tokens, output_tokens)
    usage_event = dict(
        timestamp=time.time(),
        datetime=datetime.now().astimezone().isoformat(),
        user_id=new_msg.author.id,
        channel_id=new_msg.channel.id,
        model=provider_slash_model,
        prompt_chars=prompt_chars,
        response_chars=len(response_text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        success=request_succeeded,
        error=error_summary,
    )

    try:
        await asyncio.to_thread(append_usage_event, config, usage_event)
    except Exception:
        logging.exception("Error while writing usage event")

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
