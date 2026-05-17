import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import math
import os
import re
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
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
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
    peak_active_generations: int = 0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None


monitor_state = MonitorState()
user_request_times: dict[int, list[float]] = {}
user_last_request_time: dict[int, float] = {}


def get_admin_ids(curr_config: dict[str, Any]) -> set[int]:
    return set(curr_config["permissions"]["users"]["admin_ids"])


def user_is_admin(user_id: int, curr_config: dict[str, Any]) -> bool:
    return user_id in get_admin_ids(curr_config)


def get_response_limit_kwargs(curr_config: dict[str, Any]) -> dict[str, int]:
    if max_output_tokens := curr_config.get("max_output_tokens"):
        return dict(max_tokens=int(max_output_tokens))
    return {}


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


def format_number(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def format_meter(value: float, total: float, width: int = 12) -> str:
    if total <= 0:
        return "░" * width

    filled = min(width, max(0, round(width * value / total)))
    return "█" * filled + "░" * (width - filled)


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
    provider_cost: Optional[float] = None,
) -> tuple[int, int, float]:
    cost_config = curr_config.get("cost_estimation") or {}
    input_token_count = input_tokens if input_tokens is not None else estimate_tokens(prompt_chars, float(cost_config.get("input_tokens_per_char", 0.25)))
    output_token_count = output_tokens if output_tokens is not None else estimate_tokens(response_chars, float(cost_config.get("output_tokens_per_char", 0.25)))
    estimated_cost = provider_cost if provider_cost is not None else calculate_cost(curr_config, model_name, input_token_count, output_token_count)
    return input_token_count, output_token_count, estimated_cost


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


def summarize_token_windows(events: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    now = time.time()
    windows = {
        "1h": SECONDS_PER_HOUR,
        "1w": SECONDS_PER_WEEK,
        "30d": SECONDS_PER_30_DAYS,
    }

    summaries = {}
    for label, seconds in windows.items():
        cutoff = now - seconds
        input_tokens = 0
        output_tokens = 0

        for event in events:
            if float(event.get("timestamp", 0)) < cutoff:
                continue

            input_tokens += int(event.get("input_tokens", 0))
            output_tokens += int(event.get("output_tokens", 0))

        summaries[label] = (input_tokens, output_tokens)

    return summaries


def summarize_top_messengers(events: list[dict[str, Any]], limit: int = 5) -> list[tuple[int, int, float]]:
    users = {}

    for event in events:
        if not event.get("success"):
            continue

        user_id = int(event["user_id"])
        count, cost = users.get(user_id, (0, 0.0))
        users[user_id] = (count + 1, cost + float(event.get("estimated_cost", 0)))

    return sorted(((user_id, count, cost) for user_id, (count, cost) in users.items()), key=lambda item: (-item[1], -item[2], item[0]))[:limit]


def summarize_top_spenders(events: list[dict[str, Any]], limit: int = 5) -> list[tuple[int, float, int]]:
    users = {}

    for event in events:
        if not event.get("success"):
            continue

        user_id = int(event["user_id"])
        cost, count = users.get(user_id, (0.0, 0))
        users[user_id] = (cost + float(event.get("estimated_cost", 0)), count + 1)

    return sorted(((user_id, cost, count) for user_id, (cost, count) in users.items()), key=lambda item: (-item[1], -item[2], item[0]))[:limit]


def summarize_demand(events: list[dict[str, Any]]) -> dict[str, float]:
    now = time.time()
    last_5m = [event for event in events if float(event.get("timestamp", 0)) >= now - 5 * 60]
    last_1h = [event for event in events if float(event.get("timestamp", 0)) >= now - SECONDS_PER_HOUR]
    successful_1h = [event for event in last_1h if event.get("success")]

    cost_1h = sum(float(event.get("estimated_cost", 0)) for event in last_1h)
    tokens_1h = sum(int(event.get("input_tokens", 0)) + int(event.get("output_tokens", 0)) for event in successful_1h)
    prompt_chars_1h = sum(int(event.get("prompt_chars", 0)) for event in successful_1h)
    response_chars_1h = sum(int(event.get("response_chars", 0)) for event in successful_1h)
    durations_1h = [float(event.get("duration_seconds", 0)) for event in successful_1h if event.get("duration_seconds") is not None]
    avg_duration = sum(durations_1h) / len(durations_1h) if durations_1h else 0
    failure_rate = (sum(1 for event in last_1h if not event.get("success")) / len(last_1h) * 100) if last_1h else 0

    return dict(
        requests_5m=len(last_5m),
        requests_1h=len(last_1h),
        requests_per_minute_1h=len(last_1h) / 60,
        projected_daily_cost=cost_1h * 24,
        avg_cost_per_request=cost_1h / len(successful_1h) if successful_1h else 0,
        avg_tokens_per_request=tokens_1h / len(successful_1h) if successful_1h else 0,
        avg_prompt_chars=prompt_chars_1h / len(successful_1h) if successful_1h else 0,
        avg_response_chars=response_chars_1h / len(successful_1h) if successful_1h else 0,
        avg_duration_seconds=avg_duration,
        failure_rate_1h=failure_rate,
    )


def summarize_blocked(events: list[dict[str, Any]]) -> dict[str, Any]:
    now = time.time()
    last_1h = [event for event in events if event.get("blocked_reason") and float(event.get("timestamp", 0)) >= now - SECONDS_PER_HOUR]
    reasons = {}

    for event in last_1h:
        reason = str(event.get("blocked_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1

    top_reasons = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
    return dict(blocked_1h=len(last_1h), top_reasons=top_reasons)


def has_repeated_character_spam(text: str, threshold: int) -> bool:
    return bool(threshold > 0 and re.search(rf"(.)\1{{{threshold - 1},}}", text))


def has_repeated_word_spam(text: str, threshold: int) -> bool:
    if threshold <= 0:
        return False

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return False

    longest_run = 1
    curr_run = 1

    for prev_word, word in zip(words, words[1:]):
        curr_run = curr_run + 1 if word == prev_word else 1
        longest_run = max(longest_run, curr_run)

    return longest_run >= threshold


def get_spam_block_reason(text: str, curr_config: dict[str, Any]) -> Optional[str]:
    normalized_text = text.lower()

    for pattern in curr_config.get("blocked_request_patterns", []):
        if pattern.lower() in normalized_text:
            return "blocked_phrase"

    if has_repeated_character_spam(text, int(curr_config.get("spam_repeat_char_threshold", 0))):
        return "repeated_characters"

    if has_repeated_word_spam(text, int(curr_config.get("spam_repeat_word_threshold", 0))):
        return "repeated_words"

    return None


def reserve_user_request(user_id: int, curr_config: dict[str, Any]) -> Optional[str]:
    now = time.time()
    cooldown = float(curr_config.get("per_user_cooldown_seconds", 0))
    last_request = user_last_request_time.get(user_id)

    if cooldown > 0 and last_request and now - last_request < cooldown:
        return "cooldown"

    window = float(curr_config.get("rate_limit_window_seconds", 60))
    limit = int(curr_config.get("rate_limit_per_user", 0))
    request_times = [timestamp for timestamp in user_request_times.get(user_id, []) if now - timestamp < window]

    if limit > 0 and len(request_times) >= limit:
        user_request_times[user_id] = request_times
        return "rate_limit"

    request_times.append(now)
    user_request_times[user_id] = request_times
    user_last_request_time[user_id] = now
    return None


def get_guardrail_message(reason: str) -> str:
    messages = dict(
        blocked_phrase="That request looks like a token-burner, so I’m not sending it to the model.",
        repeated_characters="That message looks like repeated-character spam, so I’m not sending it to the model.",
        repeated_words="That message looks like repeated-word spam, so I’m not sending it to the model.",
        cooldown="Slow down a bit. You’re sending requests too quickly.",
        rate_limit="You’ve hit the short-term request limit. Try again in a minute.",
        global_concurrency="The bot is already handling too many generations. Try again shortly.",
        prompt_too_large="That prompt/context is too large for this server’s bot limits.",
    )
    return messages.get(reason, "That request was blocked by the bot guardrails.")


async def reply_with_guardrail_warning(new_msg: discord.Message, reason: str) -> None:
    await new_msg.reply(get_guardrail_message(reason), silent=True)


async def log_blocked_event(curr_config: dict[str, Any], new_msg: discord.Message, model_name: str, reason: str, prompt_chars: int = 0) -> None:
    usage_event = dict(
        timestamp=time.time(),
        datetime=datetime.now().astimezone().isoformat(),
        user_id=new_msg.author.id,
        channel_id=new_msg.channel.id,
        model=model_name,
        prompt_chars=prompt_chars,
        response_chars=0,
        input_tokens=0,
        output_tokens=0,
        estimated_cost=0,
        provider_cost=None,
        duration_seconds=0,
        success=False,
        blocked_reason=reason,
        error=None,
    )

    try:
        await asyncio.to_thread(append_usage_event, curr_config, usage_event)
    except Exception:
        logging.exception("Error while writing blocked usage event")


def extract_usage(usage: Any) -> tuple[Optional[int], Optional[int], Optional[float]]:
    if usage is None:
        return None, None, None

    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    cost = getattr(usage, "cost", None)

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cost = usage.get("cost")

    return input_tokens, output_tokens, cost


async def fetch_openrouter_credits(curr_config: dict[str, Any]) -> Optional[dict[str, float]]:
    provider = curr_model.removesuffix(":vision").split("/", 1)[0]
    provider_config = curr_config["providers"].get(provider) or {}

    if "openrouter.ai" not in provider_config.get("base_url", ""):
        return None

    api_key = provider_config.get("api_key")
    if not api_key:
        return None

    response = await httpx_client.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()["data"]
    total_credits = float(data.get("total_credits", 0))
    total_usage = float(data.get("total_usage", 0))

    return dict(total_credits=total_credits, total_usage=total_usage, balance=total_credits - total_usage)


async def build_monitor_embed(curr_config: dict[str, Any]) -> discord.Embed:
    events = load_usage_events(curr_config)
    hour_cost, day_cost, month_cost = summarize_costs(events)
    top_messengers = summarize_top_messengers(events)
    top_spenders = summarize_top_spenders(events)
    demand = summarize_demand(events)
    blocked = summarize_blocked(events)
    token_windows = summarize_token_windows(events)
    openrouter_credits = None
    openrouter_error = None

    try:
        openrouter_credits = await fetch_openrouter_credits(curr_config)
    except Exception as exc:
        logging.exception("Error while fetching OpenRouter credits")
        openrouter_error = f"{type(exc).__name__}: {exc}"

    concurrency_limit = curr_config.get("global_concurrency_limit", 0)
    busy_threshold = concurrency_limit * 0.75 if concurrency_limit else float("inf")

    if monitor_state.last_error:
        health_label = "RED ERROR"
        color = discord.Color.red()
    elif monitor_state.active_generations >= busy_threshold:
        health_label = "YELLOW BUSY"
        color = discord.Color.gold()
    else:
        health_label = "GREEN HEALTHY"
        color = discord.Color.green()

    uptime = format_duration((datetime.now().astimezone() - monitor_state.start_time).total_seconds())
    latency_ms = round(discord_bot.latency * 1000) if math.isfinite(discord_bot.latency) else None
    latency_display = f"{latency_ms} ms" if latency_ms is not None else "n/a"
    success_count = max(0, monitor_state.llm_requests - monitor_state.failed_llm_requests)
    success_rate = (success_count / monitor_state.llm_requests * 100) if monitor_state.llm_requests else 100
    total_tokens_30d = sum(int(event.get("input_tokens", 0)) + int(event.get("output_tokens", 0)) for event in events)
    load_meter = format_meter(monitor_state.active_generations, concurrency_limit) if concurrency_limit else "unlimited"
    reliability_meter = format_meter(success_rate, 100)
    balance_meter = ""

    if openrouter_credits and openrouter_credits["total_credits"] > 0:
        balance_meter = format_meter(openrouter_credits["balance"], openrouter_credits["total_credits"])

    embed = discord.Embed(
        title="Bot Monitor",
        description=(
            f"**{health_label}**  |  `{curr_model}`\n"
            f"Uptime **{uptime}**  |  Gateway **{latency_display}**  |  Active **{monitor_state.active_generations}/{concurrency_limit or 'unlimited'}**"
        ),
        color=color,
        timestamp=datetime.now().astimezone(),
    )

    embed.add_field(
        name="▣ Status Overview",
        value=(
            "```text\n"
            f"Load        {load_meter} {monitor_state.active_generations}/{concurrency_limit or 'unlimited'}\n"
            f"Reliability {reliability_meter} {success_rate:.1f}%\n"
            f"Cache       {len(msg_nodes):>4} / {MAX_MESSAGE_NODES:<4}\n"
            f"Peak active {monitor_state.peak_active_generations:>4}\n"
            "```"
        ),
        inline=False,
    )
    embed.add_field(
        name="▲ Demand Pressure",
        value=(
            "```text\n"
            f"Last 5m       {demand['requests_5m']:>8.0f} req\n"
            f"Last 1h       {demand['requests_1h']:>8.0f} req\n"
            f"Rate          {demand['requests_per_minute_1h']:>8.2f}/min\n"
            f"Blocked 1h    {blocked['blocked_1h']:>8.0f}\n"
            f"24h burn      {format_money(demand['projected_daily_cost']):>8}\n"
            "```"
        ),
        inline=True,
    )
    embed.add_field(
        name="◆ Traffic",
        value=(
            "```text\n"
            f"Messages      {monitor_state.messages_seen:>8,}\n"
            f"LLM calls     {monitor_state.llm_requests:>8,}\n"
            f"Replies       {monitor_state.generated_messages:>8,}\n"
            f"Failures      {monitor_state.failed_llm_requests:>8,}\n"
            "```"
        ),
        inline=True,
    )
    embed.add_field(
        name="■ Reliability",
        value=(
            "```text\n"
            f"Success       {success_rate:>7.1f}%\n"
            f"1h failures   {demand['failure_rate_1h']:>7.1f}%\n"
            f"30d tokens    {total_tokens_30d:>8,}\n"
            "```"
        ),
        inline=True,
    )
    embed.add_field(
        name="◈ Request Shape, 1h",
        value=(
            "```text\n"
            f"Avg cost      {format_money(demand['avg_cost_per_request']):>8}\n"
            f"Avg tokens    {format_number(demand['avg_tokens_per_request']):>8}\n"
            f"Avg time      {demand['avg_duration_seconds']:>7.1f}s\n"
            "```"
        ),
        inline=True,
    )
    embed.add_field(
        name="▥ Token Flow",
        value="```text\n"
        + "\n".join(
            f"{label:<3}  in {input_tokens:>10,}   out {output_tokens:>10,}"
            for label, (input_tokens, output_tokens) in token_windows.items()
        )
        + "\n```",
        inline=False,
    )
    embed.add_field(
        name="$ Spend Windows",
        value=(
            "```text\n"
            f"Last hour     {format_money(hour_cost):>8}\n"
            f"Last day      {format_money(day_cost):>8}\n"
            f"Last 30 days  {format_money(month_cost):>8}\n"
            "```"
        ),
        inline=True,
    )

    if openrouter_credits:
        embed.add_field(
            name="◎ OpenRouter Account",
            value=(
                "```text\n"
                f"Balance      {format_money(openrouter_credits['balance']):>8}\n"
                f"Credits      {format_money(openrouter_credits['total_credits']):>8}\n"
                f"Used         {format_money(openrouter_credits['total_usage']):>8}\n"
                f"{balance_meter}\n"
                "```"
            ),
            inline=True,
        )
    elif openrouter_error:
        embed.add_field(name="◎ OpenRouter Account", value=f"Unavailable: `{openrouter_error[:180]}`", inline=True)

    leaderboard = "\n".join(
        f"`{index}.` <@{user_id}>  **{count}** req  **{format_money(cost)}**"
        for index, (user_id, count, cost) in enumerate(top_messengers, start=1)
    )
    embed.add_field(name="★ Top Messengers, 30d", value=leaderboard or "No usage yet", inline=True)

    spend_leaderboard = "\n".join(
        f"`{index}.` <@{user_id}>  **{format_money(cost)}**  **{count}** req"
        for index, (user_id, cost, count) in enumerate(top_spenders, start=1)
    )
    embed.add_field(name="$ Top Spenders, 30d", value=spend_leaderboard or "No usage yet", inline=True)

    if blocked["top_reasons"]:
        blocked_reasons = "\n".join(f"`{reason}`: **{count}**" for reason, count in blocked["top_reasons"])
        embed.add_field(name="! Top Blocks, 1h", value=blocked_reasons, inline=True)

    if monitor_state.last_error:
        embed.add_field(name="! Last Error", value=monitor_state.last_error[:1024], inline=False)

    embed.set_footer(text="OpenRouter totals are live account credits; local windows come from usage.jsonl")
    return embed


@discord_bot.tree.command(name="monitor", description="View bot health, usage, cost, and top messengers")
async def monitor_command(interaction: discord.Interaction) -> None:
    curr_config = await asyncio.to_thread(get_config)

    if not user_is_admin(interaction.user.id, curr_config):
        await interaction.response.send_message("You don't have permission to view bot monitoring.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    await interaction.followup.send(embed=await build_monitor_embed(curr_config), ephemeral=True)


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

    if not user_is_admin:
        cleaned_new_msg_content = new_msg.content.removeprefix(discord_bot.user.mention).lstrip()
        if block_reason := get_spam_block_reason(cleaned_new_msg_content, config):
            await reply_with_guardrail_warning(new_msg, block_reason)
            await log_blocked_event(config, new_msg, provider_slash_model, block_reason, len(cleaned_new_msg_content))
            return

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
    provider_cost = None
    request_succeeded = False
    error_summary = None

    if max_prompt_chars := config.get("max_prompt_chars"):
        if prompt_chars > int(max_prompt_chars):
            await reply_with_guardrail_warning(new_msg, "prompt_too_large")
            await log_blocked_event(config, new_msg, provider_slash_model, "prompt_too_large", prompt_chars)
            return

    if concurrency_limit := config.get("global_concurrency_limit"):
        if monitor_state.active_generations >= int(concurrency_limit):
            await reply_with_guardrail_warning(new_msg, "global_concurrency")
            await log_blocked_event(config, new_msg, provider_slash_model, "global_concurrency", prompt_chars)
            return

    if not user_is_admin:
        if block_reason := reserve_user_request(new_msg.author.id, config):
            await reply_with_guardrail_warning(new_msg, block_reason)
            await log_blocked_event(config, new_msg, provider_slash_model, block_reason, prompt_chars)
            return

    openai_kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=True,
        stream_options=dict(include_usage=True),
        **get_response_limit_kwargs(config),
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
    monitor_state.peak_active_generations = max(monitor_state.peak_active_generations, monitor_state.active_generations)
    monitor_state.last_request_time = datetime.now().astimezone()
    request_started_at = time.monotonic()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                chunk_input_tokens, chunk_output_tokens, chunk_cost = extract_usage(getattr(chunk, "usage", None))
                input_tokens = chunk_input_tokens if chunk_input_tokens is not None else input_tokens
                output_tokens = chunk_output_tokens if chunk_output_tokens is not None else output_tokens
                provider_cost = chunk_cost if chunk_cost is not None else provider_cost

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
    duration_seconds = time.monotonic() - request_started_at
    input_tokens, output_tokens, estimated_cost = estimate_usage_cost(config, provider_slash_model, prompt_chars, len(response_text), input_tokens, output_tokens, provider_cost)
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
        provider_cost=provider_cost,
        duration_seconds=duration_seconds,
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
