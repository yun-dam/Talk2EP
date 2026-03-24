from datasets import load_dataset

qas = ["qa11", "qa12", "qa13", "qa14", "qa15", "qa16", "qa17", "qa18", "qa19", "qa20"]
token_lens = ["128k", "256k", "512k", "1M"]

for qa in qas:
    for token_len in token_lens:
        babilong = load_dataset("RMT-team/babilong", token_len)[qa]
        babilong.to_json(f"babilong_data/babilong_{qa}_{token_len}.json")
