import time

from openai import OpenAI

_OPENAI_API_KEY=None # provide your API Key here


def chat_with_gpt(msg, model='gpt-4o', temperature=0., max_tokens=20, stream=False):
    client = OpenAI(
        api_key=_OPENAI_API_KEY,
    )
    outputs = None
    try_count = 0
    start = time.time()
    while outputs is None:
        if try_count > 5:
            print(f"Stop trying after {try_count} tries and" f" {time.time() - start:.2f} seconds.")
            return None
        try:
            try_count += 1
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            outputs = res.choices[0].message.content
        except Exception as e:
            print(f"Error message: {str(e)}")
            print("OpenAI Rate Limit reached. Sleeping for 5 minutes.")
            time.sleep(300)
    if try_count > 1:
        print(f"exited while loop after {try_count} tries and"
              f" {time.time() - start:.2f} seconds")
    
    return outputs