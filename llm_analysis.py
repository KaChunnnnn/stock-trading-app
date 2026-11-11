import requests
import json

def generate_strategy_comment(backtest_data: dict, api_key: str) -> str:
    """
    Generate AI interpretation for stock strategy back-test result.
    Args:
        backtest_data: dict from backend (portfolio_summary, stock_statistics, trade_records)
        api_key: HKBU GenAI API key input from frontend UI, DO NOT hard-code key
    Returns:
        str: AI analysis content plus mandatory disclaimer / error hint text
    """

    prompt_text = f"""
You act as a data analytics assistant. Only perform objective analysis of back-test statistics.
DO NOT provide any investment advice, buy / sell recommendations or trading guidance.

Back-test dataset:
{json.dumps(backtest_data, indent=2, ensure_ascii=False)}

Please analyse these points concisely:
1. Overall portfolio average cumulative return performance
2. Win rate and profit/loss situation for each individual stock
3. Risk evaluation based on risk metrics
4. Summary characteristics of trading behaviours
Keep output short and easy to understand.
"""

    api_endpoint_url = "https://genai.hkbu.edu.hk/api/v1/chat/completions"
    genai_model_name = "gpt-6-luna"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    request_payload = {
        "model": genai_model_name,
        "messages": [
            {"role": "user", "content": prompt_text}
        ]
    }

    try:
        response = requests.post(
            api_endpoint_url,
            json=request_payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        if response.status_code == 401:
            return "[ERROR] Invalid API key. Please check your GenAI access key."
        if response.status_code == 429:
            return "[ERROR] API rate / quota exhausted. Please try again later."

        response_json = response.json()
        ai_reply_content = response_json["choices"][0]["message"]["content"]

        disclaimer_text = "\n\nDISCLAIMER: This output is for course data-analysis demonstration only. NOT investment advice. Do NOT use it for real-world trading."
        final_output = ai_reply_content + disclaimer_text
        return final_output

    except requests.exceptions.Timeout:
        return "[ERROR] Network request timed out. Please retry."
    except Exception as general_err:
        return f"[SYSTEM FAILURE] Failed to invoke LLM API: {str(general_err)}"


def generate_llm_insight(api_key: str, stats_res, trades_res, strategy_choice: str) -> str:
    backtest_data = {
        "strategy": strategy_choice,
        "stock_statistics": stats_res.to_dict("records") if not stats_res.empty else [],
        "trade_records": trades_res.to_dict("records") if not trades_res.empty else []
    }
    return generate_strategy_comment(backtest_data, api_key)


# ---------------- LOCAL DEBUG ONLY ----------------
if __name__ == "__main__":
    from test_sample import sample_backtest_input
    test_api_key = "08173051-7a2a-4997-8ee2-998a5370654f"
    ai_result = generate_strategy_comment(sample_backtest_input, test_api_key)
    print(ai_result)