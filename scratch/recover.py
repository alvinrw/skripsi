import json

file_path = r'C:\Users\alvin\.gemini\antigravity-ide\brain\911cdfcc-f664-4be1-9ba9-02cf78ff3085\.system_generated\logs\transcript_full.jsonl'
content = None

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'tool_calls' in data and len(data['tool_calls']) > 0:
                call = data['tool_calls'][0]
                if call['name'] == 'default_api:write_to_file':
                    args = call.get('args', {})
                    if 'prepare_dataset.py' in args.get('TargetFile', ''):
                        content = args.get('CodeContent')
        except Exception as e:
            pass

if content:
    with open('src/prepare_dataset.py', 'w', encoding='utf-8') as out:
        out.write(content)
    print("prepare_dataset.py recovered successfully!")
else:
    print("Could not find prepare_dataset.py content in transcript.")
